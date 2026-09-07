import fitz  # PyMuPDF
import re
import os
import time

from project_config import PROJECT_ROOT


EXTRACTION_VERSION = "university_content_v2"
DEBUG_FOLDER = os.path.join(PROJECT_ROOT, "debug_extractions")

# =========================
# LAYOUT HANDLING
# =========================


def is_two_column(blocks, page_width):
    """Nhận diện nếu trang có cấu trúc 2 cột."""
    centers = [(b[0] + b[2]) / 2 for b in blocks]
    left = [c for c in centers if c < page_width / 2]
    right = [c for c in centers if c >= page_width / 2]
    return len(left) > 3 and len(right) > 3


def sort_blocks(blocks, page_width):
    """Sắp xếp các khối văn bản theo đúng thứ tự đọc (hỗ trợ 2 cột)."""
    if is_two_column(blocks, page_width):
        left = [b for b in blocks if (b[0] + b[2]) / 2 < page_width / 2]
        right = [b for b in blocks if (b[0] + b[2]) / 2 >= page_width / 2]

        left_sorted = sorted(left, key=lambda b: (b[1], b[0]))
        right_sorted = sorted(right, key=lambda b: (b[1], b[0]))
        return left_sorted + right_sorted
    else:
        return sorted(blocks, key=lambda b: (b[1], b[0]))


def extract_blocks_with_recovered_spaces(page):
    raw = page.get_text("rawdict")
    recovered_blocks = []
    closing_punctuation = set(".,;:!?%)]}")
    opening_punctuation = set("([{")

    for block_number, block in enumerate(raw.get("blocks", [])):
        if block.get("type") != 0:
            continue
        line_texts = []
        for line in block.get("lines", []):
            characters = []
            for span in line.get("spans", []):
                font_size = float(span.get("size") or 0.0)
                for character in span.get("chars", []):
                    characters.append((character, font_size))
            if not characters:
                continue

            output = []
            for index, (character, font_size) in enumerate(characters):
                value = character.get("c", "")
                output.append(value)
                if index + 1 >= len(characters):
                    continue

                next_character, next_font_size = characters[index + 1]
                next_value = next_character.get("c", "")
                if (
                    not value
                    or not next_value
                    or value.isspace()
                    or next_value.isspace()
                    or next_value in closing_punctuation
                    or value in opening_punctuation
                ):
                    continue

                gap = (
                    float(next_character["bbox"][0])
                    - float(character["bbox"][2])
                )
                average_font_size = max(
                    (font_size + next_font_size) / 2.0, 1.0)
                minimum_word_gap = max(1.25, average_font_size * 0.15)
                if gap > minimum_word_gap:
                    output.append(" ")

            line_texts.append("".join(output).strip())

        text = "\n".join(line for line in line_texts if line)
        if not text:
            continue
        bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
        recovered_blocks.append((
            bbox[0], bbox[1], bbox[2], bbox[3], text, block_number, 0,
        ))
    return recovered_blocks


def find_abstract_start_page(doc, max_scan_pages=6):
    """
    Thay vì cắt cứng 3 trang đầu, tìm trang có chứa 'Abstract' / 'Tóm tắt'
    để bắt đầu trích xuất từ đó. Nếu không tìm thấy, trả về 0 (lấy từ đầu).
    Tránh việc bỏ sót nội dung ở các bài báo ngắn (4-6 trang).
    """
    marker_pattern = re.compile(
        r"(?im)^\s*(abstract|tóm tắt)\s*[:.]?\s*$")
    scan_limit = min(max_scan_pages, doc.page_count)
    for i in range(scan_limit):
        page_text = doc[i].get_text("text")
        if marker_pattern.search(page_text):
            return i
    return 0


def has_reference_heading(text):
    return bool(re.search(
        r"(?im)^\s*(references|bibliography|tài liệu tham khảo)\s*[:.]?\s*$",
        text or "",
    ))


def split_before_reference_heading(text):
    offset = 0
    for line in (text or "").splitlines(keepends=True):
        if has_reference_heading(line):
            return text[:offset].rstrip(), True
        offset += len(line)
    return text, False


# =========================
# NOISE REMOVAL (Layout-level: page numbers, TOC, running headers)
# =========================


def remove_page_noise(text):
    """
    Loại bỏ nhiễu cấp độ TRANG: số trang đứng riêng, dòng Mục lục (TOC),
    ligature lạ. KHÔNG xóa dấu câu, KHÔNG lowercase — giữ nguyên cấu trúc
    câu để phục vụ chunking & embedding ở bước sau.
    """
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
    lines = text.split("\n")

    cleaned_lines = []
    for l in lines:
        stripped = l.strip()

        # Dòng chỉ chứa số trang (vd: "12")
        if re.fullmatch(r'\d+', stripped):
            continue

        # Dòng Mục lục kiểu "Chapter 1 ..... 12"
        if re.search(r'\.{4,}\s*\d+$', stripped):
            continue

        # Dòng chứa quá nhiều số rời rạc (thường là bảng số liệu / mục lục)
        if len(re.findall(r'\d+', stripped)) > 6 and len(stripped) < 120:
            continue

        cleaned_lines.append(l)

    text = "\n".join(cleaned_lines)

    # Caption hình/bảng — vẫn nên loại vì không phải nội dung văn xuôi cần so khớp đạo văn
    text = re.sub(
        r'(?im)^\s*(fig(?:ure)?|table|hình|bảng)\.?\s*\d+[:.\-]?.*$', ' ', text)

    # Gộp các dòng bị ngắt giữa câu (do PDF wrap) thành 1 dòng,
    # nhưng giữ lại xuống dòng đôi (ranh giới đoạn văn)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def normalize_for_matching(text):
    """
    Bước chuẩn hoá MẠNH dùng riêng cho TF-IDF (bag-of-words).
    Lowercase + xoá dấu câu. KHÔNG dùng bước này làm input cho
    chunker/embedding vì sẽ phá vỡ ranh giới câu.
    """
    text = text.lower()
    text = re.sub(r'\b\d+(\.\d+)+\b', ' ', text)  # số mục lục kiểu 1.2.3
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================
# MAIN INTERFACE
# =========================


def extract_university_content(input_data, filename=None, save_debug=True):
    """
    Trích xuất văn bản sạch (giữ câu, giữ dấu câu) từ PDF hoặc TXT.
    Trả về plain text đã loại bỏ header/footer/số trang/mục lục/references,
    PHÙ HỢP để đưa thẳng vào chunker + model embedding.

    Nếu cần bản chuẩn hoá mạnh cho TF-IDF, gọi thêm normalize_for_matching().
    """
    doc = None
    final_text = ""
    debug_name = filename if filename else f"extracted_{int(time.time())}.txt"
    if not debug_name.endswith(".txt"):
        debug_name = os.path.splitext(debug_name)[0] + ".txt"

    try:
        # --- CASE 1: INPUT IS BYTES (FastAPI) ---
        if isinstance(input_data, bytes):
            if filename and filename.lower().endswith(".txt"):
                final_text = remove_page_noise(
                    input_data.decode('utf-8-sig', errors='ignore'))
            else:
                doc = fitz.open(stream=input_data, filetype="pdf")

        # --- CASE 2: INPUT IS STRING PATH ---
        elif isinstance(input_data, str):
            if not os.path.exists(input_data):
                return ""
            if input_data.lower().endswith(".txt"):
                with open(input_data, 'r', encoding='utf-8', errors='ignore') as f:
                    final_text = remove_page_noise(f.read())
            else:
                doc = fitz.open(input_data)

        # --- XỬ LÝ PDF NẾU CẦN ---
        if doc:
            all_pages_text = []
            total_pages = doc.page_count

            # Thay vì cắt cứng 3 trang đầu, tìm trang Abstract thực sự.
            start_page = find_abstract_start_page(doc)

            # Cắt bỏ References — quét từ cuối lên, không lùi quá start_page
            cutoff_page = total_pages
            for i in range(total_pages - 1, start_page - 1, -1):
                page_peek = doc[i].get_text("text")
                if has_reference_heading(page_peek):
                    cutoff_page = i
                    break

            last_page = min(cutoff_page + 1, total_pages)
            for i in range(start_page, last_page):
                page = doc[i]
                blocks = extract_blocks_with_recovered_spaces(page)
                # Lọc Header/Footer (theo vị trí y trên trang) & Sắp xếp layout
                filtered = [b for b in blocks if 40 < b[1] <
                            (page.rect.height - 40) and b[4].strip()]
                sorted_blocks = sort_blocks(filtered, page.rect.width)
                page_parts = []
                for block in sorted_blocks:
                    block_text = block[4].strip()
                    if i == cutoff_page:
                        block_text, found_heading = (
                            split_before_reference_heading(block_text)
                        )
                        if block_text:
                            page_parts.append(block_text)
                        if found_heading:
                            break
                    else:
                        page_parts.append(block_text)
                all_pages_text.append("\n".join(page_parts))

            doc.close()
            final_text = remove_page_noise("\n\n".join(all_pages_text))

        # --- LƯU VÀO THƯ MỤC DEBUG ---
        if save_debug and final_text:
            os.makedirs(DEBUG_FOLDER, exist_ok=True)
            debug_path = os.path.join(DEBUG_FOLDER, debug_name)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(
                f"[*] Debug: Nội dung trích xuất đã được lưu tại {debug_path}")

        return final_text

    except Exception as e:
        if doc:
            doc.close()
        print(f"Extraction Error: {e}")
        return ""
