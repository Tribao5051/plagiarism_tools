import argparse
import hashlib
import json
import os
import re
import sys
import time
import tomllib
from urllib.parse import urljoin

import requests

from corpus_metadata import (
    METADATA_SUFFIX,
    file_sha256,
    normalize_faculty,
    write_json_file,
)
from project_config import DEFAULT_CONFIG_PATH, PROJECT_ROOT


DEFAULT_MANIFEST_NAME = "dspace_sync_manifest.json"


def safe_filename(value):
    value = os.path.basename(str(value or "document"))
    value = re.sub(r"[<>:\"/\\|?*\x00-\x1f]", "_", value).strip(" .")
    return value or "document"


def first_metadata_value(item, fields):
    metadata = item.get("metadata") or {}
    for field in fields:
        values = metadata.get(field) or []
        for entry in values:
            value = entry.get("value") if isinstance(entry, dict) else entry
            if value:
                return str(value)
    return None


def resolve_faculty(item, metadata_fields, value_map, fallback):
    raw_value = first_metadata_value(item, metadata_fields)
    if not raw_value:
        return normalize_faculty(fallback)

    configured_map = value_map or {}
    mapped_value = (
        configured_map.get(raw_value)
        or configured_map.get(raw_value.strip().casefold())
        or configured_map.get(normalize_faculty(raw_value))
    )
    return normalize_faculty(mapped_value or raw_value)


def embedded_list(payload, *path):
    current = payload
    for key in path:
        if not isinstance(current, dict):
            return []
        current = current.get(key)
    return current if isinstance(current, list) else []


def write_json_if_changed(path, payload):
    current = None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            current = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if current == payload:
        return False
    write_json_file(path, payload)
    return True


def checksum_matches(path, checksum):
    """Return True when a supported DSpace checksum matches the local file."""
    if not checksum or ":" not in checksum:
        return True
    algorithm, expected = checksum.split(":", 1)
    algorithm_name = re.sub(r"[^a-z0-9]", "", algorithm.casefold())
    aliases = {
        "md5": "md5",
        "sha1": "sha1",
        "sha256": "sha256",
        "sha512": "sha512",
    }
    digest_name = aliases.get(algorithm_name)
    if not digest_name:
        return True
    digest = hashlib.new(digest_name)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().casefold() == expected.strip().casefold()


def remove_managed_paths(output_dir, relative_path):
    if not relative_path:
        return False
    output_root = os.path.abspath(output_dir)
    target = os.path.abspath(os.path.join(
        output_root, str(relative_path).replace("/", os.sep)))
    if os.path.commonpath([output_root, target]) != output_root:
        raise ValueError(f"Refusing to remove path outside output root: {target}")
    removed = False
    for path in (target, target + METADATA_SUFFIX):
        if os.path.isfile(path):
            os.remove(path)
            removed = True
    current = os.path.dirname(target)
    while current != output_root:
        try:
            os.rmdir(current)
        except OSError:
            break
        current = os.path.dirname(current)
    return removed


class DSpaceClient:
    def __init__(self, base_url, bearer_token=None, timeout=60):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/hal+json"})
        if bearer_token:
            self.session.headers["Authorization"] = (
                f"Bearer {bearer_token}")

    def login(self, username, password):
        csrf_response = self.session.get(
            self.absolute_url("/api/security/csrf"),
            timeout=self.timeout,
        )
        csrf_response.raise_for_status()
        csrf_token = csrf_response.headers.get("DSPACE-XSRF-TOKEN")
        if not csrf_token:
            raise RuntimeError("DSpace did not return DSPACE-XSRF-TOKEN")
        response = self.session.post(
            self.absolute_url("/api/authn/login"),
            data={"user": username, "password": password},
            headers={"X-XSRF-TOKEN": csrf_token},
            timeout=self.timeout,
        )
        response.raise_for_status()
        authorization = response.headers.get("Authorization")
        if not authorization:
            raise RuntimeError("DSpace login did not return Authorization JWT")
        self.session.headers["Authorization"] = authorization

    def absolute_url(self, path_or_url):
        if str(path_or_url).startswith(("http://", "https://")):
            return str(path_or_url)
        return urljoin(self.base_url, str(path_or_url).lstrip("/"))

    def get_json(self, path_or_url, params=None):
        response = self.session.get(
            self.absolute_url(path_or_url),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def iter_embedded(self, path_or_url, embedded_key, page_size=100):
        page = 0
        while True:
            payload = self.get_json(
                path_or_url, params={"page": page, "size": page_size})
            entries = embedded_list(payload, "_embedded", embedded_key)
            yield from entries
            page_info = payload.get("page") or {}
            total_pages = int(page_info.get("totalPages") or 0)
            if not entries or page + 1 >= total_pages:
                break
            page += 1

    def list_communities(self, page_size=100):
        return list(self.iter_embedded(
            "/api/core/communities", "communities", page_size))

    def list_collections(self, community, page_size=100):
        uuid = community.get("uuid") or community.get("id")
        return list(self.iter_embedded(
            f"/api/core/communities/{uuid}/collections",
            "collections",
            page_size,
        ))

    def discover_repository_tree(self, page_size=100):
        scopes = []
        communities = []
        for community in self.list_communities(page_size):
            community_uuid = community.get("uuid") or community.get("id")
            if not community_uuid:
                continue
            community_name = community.get("name") or community_uuid
            communities.append({
                "uuid": community_uuid,
                "name": community_name,
                "parent_uuid": None,
                "path": community_name,
            })
            for collection in self.list_collections(community, page_size):
                collection_uuid = collection.get("uuid") or collection.get("id")
                collection_name = collection.get("name") or collection_uuid
                if not collection_uuid:
                    continue
                scopes.append({
                    "uuid": collection_uuid,
                    "faculty": normalize_faculty(collection_name),
                    "collection_name": collection_name,
                    "community_uuid": community_uuid,
                    "community_name": community_name,
                    "community_path": community_name,
                })
        return {
            "communities": sorted(communities, key=lambda item: (
                item["path"].casefold(), item["uuid"])),
            "collections": sorted(scopes, key=lambda item: (
                item["community_path"].casefold(),
                item["collection_name"].casefold(),
                item["uuid"],
            )),
        }

    def discover_collection_scopes(self, page_size=100):
        return self.discover_repository_tree(page_size)["collections"]

    def iter_discovery_items(self, scope=None, page_size=100):
        page = 0
        while True:
            params = {
                "page": page,
                "size": page_size,
            }
            if scope:
                params["scope"] = scope
            payload = self.get_json(
                "/api/discover/search/objects", params=params)
            objects = embedded_list(
                payload, "_embedded", "searchResult", "_embedded", "objects")
            for result in objects:
                item = (
                    result.get("_embedded", {}).get("indexableObject")
                    if isinstance(result, dict) else None
                )
                if (
                    isinstance(item, dict)
                    and item.get("type") in (None, "item")
                    and (item.get("uuid") or item.get("id"))
                ):
                    yield item

            page_info = payload.get("page") or (
                payload.get("_embedded", {})
                .get("searchResult", {})
                .get("page", {})
            )
            total_pages = int(page_info.get("totalPages") or 0)
            if not objects or page + 1 >= total_pages:
                break
            page += 1

    def get_item(self, item_uuid):
        return self.get_json(f"/api/core/items/{item_uuid}")

    def list_bundles(self, item):
        url = (
            item.get("_links", {}).get("bundles", {}).get("href")
            or f"/api/core/items/{item['uuid']}/bundles"
        )
        payload = self.get_json(url, params={"size": 100})
        return embedded_list(payload, "_embedded", "bundles")

    def list_bitstreams(self, bundle):
        url = (
            bundle.get("_links", {}).get("bitstreams", {}).get("href")
            or f"/api/core/bundles/{bundle['uuid']}/bitstreams"
        )
        payload = self.get_json(url, params={"size": 100})
        return embedded_list(payload, "_embedded", "bitstreams")

    def download(self, path_or_url, destination):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary_path = destination + ".part"
        with self.session.get(
            self.absolute_url(path_or_url),
            stream=True,
            timeout=self.timeout,
            headers={"Accept": "*/*"},
        ) as response:
            response.raise_for_status()
            with open(temporary_path, "wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
        os.replace(temporary_path, destination)


def load_config(path):
    if os.path.splitext(path)[1].lower() == ".toml":
        with open(path, "rb") as handle:
            payload = tomllib.load(handle)
        config = dict(payload.get("dspace") or {})
    else:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    if not config.get("base_url"):
        raise ValueError("dspace config requires base_url")
    config.setdefault("output_dir", "dspace_corpus")
    config.setdefault("page_size", 100)
    config.setdefault("request_timeout_seconds", 60)
    config.setdefault("scopes", [])
    config.setdefault("discover_all_collections", False)
    config.setdefault("group_by_collection_name", True)
    config.setdefault("default_faculty", "unassigned")
    config.setdefault("fetch_full_item_metadata", True)
    config.setdefault("faculty_value_map", {})
    config.setdefault("faculty_metadata_fields", [
        "local.department",
        "dc.contributor.department",
        "dc.subject.department",
    ])
    config.setdefault("allowed_extensions", [".pdf", ".txt"])
    config.setdefault("bearer_token_env", "DSPACE_BEARER_TOKEN")
    config.setdefault("username_env", "DSPACE_USERNAME")
    config.setdefault("password_env", "DSPACE_PASSWORD")
    return config


def resolve_output_dir(config, config_path):
    output_dir = config["output_dir"]
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.abspath(os.path.join(PROJECT_ROOT, output_dir))


def bitstream_checksum(bitstream):
    checksum = bitstream.get("checkSum") or {}
    value = checksum.get("value")
    algorithm = checksum.get("checkSumAlgorithm")
    return f"{algorithm}:{value}" if value else None


def resolve_sync_scopes(client, config):
    if config["discover_all_collections"]:
        return client.discover_collection_scopes(int(config["page_size"]))
    scopes = []
    for configured in config["scopes"]:
        scope = dict(configured)
        scope.setdefault("collection_name", scope.get("faculty"))
        scope.setdefault("community_uuid", None)
        scope.setdefault("community_name", None)
        scope.setdefault("community_path", None)
        scopes.append(scope)
    return scopes or [{
        "uuid": None,
        "faculty": config["default_faculty"],
        "collection_name": None,
        "community_uuid": None,
        "community_name": None,
        "community_path": None,
    }]


def sync_dspace(
    config_path=DEFAULT_CONFIG_PATH,
    dry_run=False,
    output_dir_override=None,
    max_downloads=None,
    max_per_collection=None,
):
    config = load_config(config_path)
    output_dir = (
        os.path.abspath(output_dir_override)
        if output_dir_override
        else resolve_output_dir(config, config_path)
    )
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, DEFAULT_MANIFEST_NAME)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        manifest = {"bitstreams": {}}
    manifest.setdefault("bitstreams", {})

    token = os.getenv(config["bearer_token_env"])
    client = DSpaceClient(
        config["base_url"],
        bearer_token=token,
        timeout=int(config["request_timeout_seconds"]),
    )
    if not token:
        username = os.getenv(config["username_env"])
        password = os.getenv(config["password_env"])
        if username and password:
            client.login(username, password)
    allowed_extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in config["allowed_extensions"]
    }
    if (
        config["discover_all_collections"]
        and hasattr(client, "discover_repository_tree")
    ):
        repository_tree = client.discover_repository_tree(
            int(config["page_size"]))
        scopes = repository_tree["collections"]
        communities = repository_tree["communities"]
    else:
        scopes = resolve_sync_scopes(client, config)
        communities_by_uuid = {}
        for scope in scopes:
            community_uuid = scope.get("community_uuid")
            if community_uuid:
                communities_by_uuid[community_uuid] = {
                    "uuid": community_uuid,
                    "name": scope.get("community_name"),
                    "parent_uuid": None,
                    "path": scope.get("community_path"),
                }
        communities = list(communities_by_uuid.values())
    seen_items = set()
    unique_item_uuids = set()
    seen_bitstreams = set()
    selected_files = 0
    report = {
        "output_dir": output_dir,
        "discover_all_collections": config["discover_all_collections"],
        "communities_discovered": len(communities),
        "communities": communities,
        "collections_discovered": len(scopes),
        "collection_names": sorted({
            scope.get("collection_name")
            for scope in scopes
            if scope.get("collection_name")
        }),
        "items_seen": 0,
        "bitstreams_seen": 0,
        "selected_files": 0,
        "downloaded": 0,
        "skipped_unchanged": 0,
        "metadata_updated": 0,
        "relocated": 0,
        "removed_stale": 0,
        "checksum_failures": 0,
        "unsupported_files": 0,
        "limited_files": 0,
        "errors": [],
        "faculties": {},
        "collections": {},
    }

    for scope in scopes:
        if max_downloads is not None and selected_files >= max_downloads:
            break
        scope_uuid = scope.get("uuid")
        scope_faculty = scope.get("faculty")
        collection_name = scope.get("collection_name")
        collection_folder = normalize_faculty(
            collection_name or scope_uuid or config["default_faculty"])
        faculty_fallback = (
            collection_name
            if config["group_by_collection_name"] and collection_name
            else scope_faculty or config["default_faculty"]
        )
        selected_in_collection = 0
        collection_report = report["collections"].setdefault(
            scope_uuid or "all",
            {
                "name": collection_name,
                "community_name": scope.get("community_name"),
                "community_uuid": scope.get("community_uuid"),
                "community_path": scope.get("community_path"),
                "folder": collection_folder,
                "items_seen": 0,
                "item_uuids": [],
                "selected_files": 0,
            },
        )
        for item in client.iter_discovery_items(
                scope=scope_uuid, page_size=int(config["page_size"])):
            if max_downloads is not None and selected_files >= max_downloads:
                break
            item_uuid = item.get("uuid") or item.get("id")
            item_identity = (scope_uuid, item_uuid)
            if not item_uuid or item_identity in seen_items:
                continue
            seen_items.add(item_identity)
            unique_item_uuids.add(item_uuid)
            report["items_seen"] += 1
            collection_report["items_seen"] += 1
            collection_report["item_uuids"].append(item_uuid)
            if config["fetch_full_item_metadata"]:
                try:
                    item = client.get_item(item_uuid)
                except requests.RequestException as exc:
                    report["errors"].append(
                        f"item {item_uuid} metadata: {exc}")
                    continue
            raw_faculty = first_metadata_value(
                item, config["faculty_metadata_fields"])
            faculty = resolve_faculty(
                item,
                config["faculty_metadata_fields"],
                config["faculty_value_map"],
                faculty_fallback,
            )
            report["faculties"][faculty] = (
                report["faculties"].get(faculty, 0) + 1)
            title = first_metadata_value(item, ["dc.title"]) or item.get("name")

            try:
                bundles = client.list_bundles(item)
            except requests.RequestException as exc:
                report["errors"].append(
                    f"item {item_uuid} bundles: {exc}")
                continue

            for bundle in bundles:
                if str(bundle.get("name", "")).upper() != "ORIGINAL":
                    continue
                try:
                    bitstreams = client.list_bitstreams(bundle)
                except requests.RequestException as exc:
                    report["errors"].append(
                        f"bundle {bundle.get('uuid')} bitstreams: {exc}")
                    continue

                for bitstream in bitstreams:
                    if max_downloads is not None and selected_files >= max_downloads:
                        break
                    if (
                        max_per_collection is not None
                        and selected_in_collection >= max_per_collection
                    ):
                        report["limited_files"] += 1
                        break
                    bitstream_uuid = bitstream.get("uuid") or bitstream.get("id")
                    if not bitstream_uuid or bitstream_uuid in seen_bitstreams:
                        continue
                    seen_bitstreams.add(bitstream_uuid)
                    report["bitstreams_seen"] += 1
                    original_name = safe_filename(
                        bitstream.get("name") or bitstream_uuid)
                    extension = os.path.splitext(original_name)[1].lower()
                    if extension not in allowed_extensions:
                        report["unsupported_files"] += 1
                        continue

                    selected_files += 1
                    selected_in_collection += 1
                    report["selected_files"] = selected_files
                    collection_report["selected_files"] += 1

                    local_name = f"{bitstream_uuid}_{original_name}"
                    storage_folder = (
                        collection_folder
                        if config["group_by_collection_name"]
                        else faculty
                    )
                    destination = os.path.join(
                        output_dir, storage_folder, item_uuid, local_name)
                    checksum = bitstream_checksum(bitstream)
                    previous = manifest["bitstreams"].get(bitstream_uuid, {})
                    content_url = (
                        bitstream.get("_links", {}).get("content", {}).get("href")
                        or f"/api/core/bitstreams/{bitstream_uuid}/content"
                    )
                    relative_destination = os.path.relpath(
                        destination, output_dir).replace("\\", "/")
                    candidate_path = destination
                    previous_relative_path = previous.get("path")
                    if previous_relative_path and not os.path.exists(
                            candidate_path):
                        previous_path = os.path.abspath(os.path.join(
                            output_dir,
                            previous_relative_path.replace("/", os.sep),
                        ))
                        output_root = os.path.abspath(output_dir)
                        if (
                            os.path.commonpath(
                                [output_root, previous_path]) == output_root
                            and os.path.isfile(previous_path)
                        ):
                            candidate_path = previous_path
                    candidate_unchanged = (
                        os.path.exists(candidate_path)
                        and previous.get("checksum") == checksum
                    )
                    local_sha256 = None
                    unchanged = False
                    if candidate_unchanged:
                        local_sha256 = file_sha256(candidate_path)
                        unchanged = (
                            checksum_matches(candidate_path, checksum)
                            and (
                                not previous.get("local_sha256")
                                or previous["local_sha256"] == local_sha256
                            )
                        )

                    sidecar = {
                        "faculty": faculty,
                        "dspace_faculty_value": raw_faculty,
                        "title": title or original_name,
                        "dspace_item_uuid": item_uuid,
                        "dspace_bitstream_uuid": bitstream_uuid,
                        "dspace_handle": item.get("handle"),
                        "dspace_collection_uuid": scope_uuid,
                        "dspace_collection_name": collection_name,
                        "dspace_collection_folder": collection_folder,
                        "dspace_community_uuid": scope.get("community_uuid"),
                        "dspace_community_name": scope.get("community_name"),
                        "dspace_community_path": scope.get("community_path"),
                        "dspace_last_modified": item.get("lastModified"),
                        "dspace_checksum": checksum,
                        "dspace_source_url": content_url,
                        "dspace_metadata": item.get("metadata") or {},
                    }
                    if dry_run:
                        report[
                            "skipped_unchanged" if unchanged else "downloaded"
                        ] += 1
                        continue

                    try:
                        if unchanged:
                            if os.path.abspath(candidate_path) != os.path.abspath(
                                    destination):
                                os.makedirs(
                                    os.path.dirname(destination), exist_ok=True)
                                os.replace(candidate_path, destination)
                            report["skipped_unchanged"] += 1
                        else:
                            client.download(content_url, destination)
                            if not checksum_matches(destination, checksum):
                                report["checksum_failures"] += 1
                                os.remove(destination)
                                raise OSError(
                                    f"checksum mismatch: expected {checksum}")
                            local_sha256 = file_sha256(destination)
                            report["downloaded"] += 1
                        sidecar["dspace_file_sha256"] = local_sha256
                        if write_json_if_changed(
                                destination + METADATA_SUFFIX, sidecar):
                            report["metadata_updated"] += 1
                        if previous_relative_path:
                            previous_path = os.path.abspath(os.path.join(
                                output_dir,
                                previous_relative_path.replace("/", os.sep),
                            ))
                            output_root = os.path.abspath(output_dir)
                            if (
                                previous_path != os.path.abspath(destination)
                                and os.path.commonpath(
                                    [output_root, previous_path]) == output_root
                            ):
                                removed = remove_managed_paths(
                                    output_dir, previous_relative_path)
                                if removed:
                                    report["relocated"] += 1
                        manifest["bitstreams"][bitstream_uuid] = {
                            "checksum": checksum,
                            "local_sha256": local_sha256,
                            "size_bytes": os.path.getsize(destination),
                            "path": relative_destination,
                            "faculty": faculty,
                            "item_uuid": item_uuid,
                            "collection_uuid": scope_uuid,
                            "collection_name": collection_name,
                            "collection_folder": collection_folder,
                            "community_uuid": scope.get("community_uuid"),
                            "community_name": scope.get("community_name"),
                        }
                    except (OSError, requests.RequestException) as exc:
                        report["errors"].append(
                            f"bitstream {bitstream_uuid}: {exc}")

    full_sync = max_downloads is None and max_per_collection is None
    if full_sync and not dry_run and not report["errors"]:
        current_scope_uuids = {
            scope.get("uuid") for scope in scopes if scope.get("uuid")
        }
        authoritative_all = bool(config["discover_all_collections"])
        stale_bitstreams = []
        for bitstream_uuid, previous in manifest["bitstreams"].items():
            if bitstream_uuid in seen_bitstreams:
                continue
            if (
                authoritative_all
                or previous.get("collection_uuid") in current_scope_uuids
            ):
                stale_bitstreams.append(bitstream_uuid)
        for bitstream_uuid in stale_bitstreams:
            previous = manifest["bitstreams"].pop(bitstream_uuid)
            if remove_managed_paths(output_dir, previous.get("path")):
                report["removed_stale"] += 1

    report["unique_items_seen"] = len(unique_item_uuids)
    report["unique_item_uuids"] = sorted(unique_item_uuids)
    report["finished_at_unix"] = time.time()
    if not dry_run:
        manifest["last_sync_unix"] = report["finished_at_unix"]
        write_json_if_changed(manifest_path, manifest)
        write_json_file(
            os.path.join(output_dir, "dspace_sync_report.json"), report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download DSpace ORIGINAL PDF/TXT bitstreams by faculty.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-downloads", type=int, default=None)
    parser.add_argument("--max-per-collection", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, OSError):
        pass
    report = sync_dspace(
        args.config,
        dry_run=args.dry_run,
        output_dir_override=args.output_dir,
        max_downloads=args.max_downloads,
        max_per_collection=args.max_per_collection,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
