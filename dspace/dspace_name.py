import requests


# ========================== CONFIGURATION ==========================
BASE_URL = "http://dspace.lib.test/server"

# Leave empty if the data is public.
ACCESS_TOKEN = ""

PAGE_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 30
VERIFY_SSL = True
# ===================================================================


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    if ACCESS_TOKEN.strip():
        session.headers["Authorization"] = f"Bearer {ACCESS_TOKEN.strip()}"

    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
) -> dict:
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        verify=VERIFY_SSL,
    )
    response.raise_for_status()
    return response.json()


def get_all_hal_objects(
    session: requests.Session,
    endpoint: str,
    embedded_key: str,
) -> list[dict]:
    """Read every page from a regular DSpace HAL endpoint."""
    objects = []
    page_number = 0

    while True:
        body = get_json(
            session,
            f"{BASE_URL.rstrip('/')}{endpoint}",
            {
                "page": page_number,
                "size": PAGE_SIZE,
            },
        )

        objects.extend(
            body.get("_embedded", {}).get(embedded_key, [])
        )

        page = body.get("page", {})
        total_pages = int(page.get("totalPages", 1))

        page_number += 1
        if page_number >= total_pages:
            break

    return objects


def get_all_submissions(
    session: requests.Session,
    collection_uuid: str,
) -> list[dict]:
    """Read all archived/discoverable Items in one Collection."""
    submissions = []
    page_number = 0
    url = f"{BASE_URL.rstrip('/')}/api/discover/search/objects"

    while True:
        body = get_json(
            session,
            url,
            {
                "scope": collection_uuid,
                "dsoType": "item",
                "page": page_number,
                "size": PAGE_SIZE,
            },
        )

        search_result = (
            body.get("_embedded", {})
            .get("searchResult", {})
        )
        results = (
            search_result.get("_embedded", {})
            .get("objects", [])
        )

        for result in results:
            item = (
                result.get("_embedded", {})
                .get("indexableObject")
            )
            if item:
                submissions.append(item)

        page = search_result.get("page", {})
        total_pages = int(page.get("totalPages", 1))

        page_number += 1
        if page_number >= total_pages:
            break

    return submissions


def get_direct_collections_of_communities(
    session: requests.Session,
    communities: list[dict],
) -> list[dict]:
    """Read only Collections directly linked to the given Communities."""
    collections = []

    for community in communities:
        community_uuid = community.get("uuid")
        if not community_uuid:
            continue

        collections.extend(
            get_all_hal_objects(
                session,
                f"/api/core/communities/{community_uuid}/collections",
                "collections",
            )
        )

    return collections


def metadata_values(item: dict, field: str) -> list[str]:
    return [
        entry.get("value", "").strip()
        for entry in item.get("metadata", {}).get(field, [])
        if entry.get("value", "").strip()
    ]


def submission_name(item: dict) -> str:
    titles = metadata_values(item, "dc.title")
    if titles:
        return titles[0]
    return item.get("name") or "Không có tên đề tài"


def submission_authors(item: dict) -> str:
    authors = metadata_values(item, "dc.contributor.author")
    return "; ".join(authors) if authors else "Không có thông tin tác giả"


def main() -> None:
    with build_session() as session:
        communities = get_all_hal_objects(
            session,
            "/api/core/communities/search/top",
            "communities",
        )
        collections = get_direct_collections_of_communities(
            session,
            communities,
        )

        # Read submissions from every direct Collection. Use UUID as the key
        # so an Item mapped into multiple Collections is only counted once.
        submissions_by_uuid = {}
        for collection in collections:
            collection_uuid = collection.get("uuid")
            if not collection_uuid:
                continue

            for item in get_all_submissions(session, collection_uuid):
                item_key = item.get("uuid") or item.get("handle")
                if item_key:
                    submissions_by_uuid.setdefault(item_key, item)

        submissions = list(submissions_by_uuid.values())

    print(f"Số Community top-level: {len(communities)}")
    print(f"Số Collection trực tiếp: {len(collections)}")
    print(f"Số Submission: {len(submissions)}")
    print()

    for index, item in enumerate(submissions, start=1):
        print(f"{index}. {submission_name(item)}")
        print(f"   Tác giả: {submission_authors(item)}")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as error:
        response = error.response
        print(f"Lỗi HTTP {response.status_code}: {response.text}")
    except requests.RequestException as error:
        print(f"Không thể kết nối tới DSpace: {error}")
