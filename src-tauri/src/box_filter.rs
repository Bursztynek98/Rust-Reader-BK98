//! OCR bounding box and character garbage filtering.
//! Rejects OCR noise, Chinese/CJK/Cyrillic characters, standalone dots/commas, and repetitive character spam.

pub fn is_valid_box_size(w: u32, h: u32, _frame_w: u32, frame_h: u32) -> bool {
    // Filter tiny noise boxes (< 8px height or < 1.5% frame height)
    let min_h = 8u32.max((frame_h as f32 * 0.015) as u32);
    if h < min_h || w < 8 {
        return false;
    }

    // Filter giant full-screen artifact boxes (> 92% frame height)
    let max_h = (frame_h as f32 * 0.92) as u32;
    if h > max_h {
        return false;
    }

    true
}

/// Check if a character is a valid Latin or Polish alphabet character, digit, or basic space/punctuation
pub fn is_allowed_char(c: char) -> bool {
    match c {
        'a'..='z' | 'A'..='Z' | '0'..='9' => true,
        'ą' | 'ć' | 'ę' | 'ł' | 'ń' | 'ó' | 'ś' | 'ż' | 'ź' => true,
        'Ą' | 'Ć' | 'Ę' | 'Ł' | 'Ń' | 'Ó' | 'Ś' | 'Ż' | 'Ź' => true,
        ' ' | '.' | ',' | '!' | '?' | '-' | '\'' | '"' | ':' | ';' | '(' | ')' => true,
        _ => false, // Rejects CJK Chinese/Japanese, Cyrillic, Box drawing, symbols
    }
}

/// Strip out any invalid non-Latin/CJK characters from text
pub fn sanitize_text(text: &str) -> String {
    text.chars().filter(|&c| is_allowed_char(c)).collect::<String>()
}

/// Checks if text is valid human-readable Polish/Latin text and not punctuation/Chinese/garbage noise.
pub fn is_clean_text(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }

    // 1. MUST contain at least one valid Polish/Latin letter (rejects pure dots, commas, symbols, numbers only)
    let has_at_least_one_letter = trimmed.chars().any(|c| {
        matches!(c, 'a'..='z' | 'A'..='Z' | 'ą' | 'ć' | 'ę' | 'ł' | 'ń' | 'ó' | 'ś' | 'ż' | 'ź' | 'Ą' | 'Ć' | 'Ę' | 'Ł' | 'Ń' | 'Ó' | 'Ś' | 'Ż' | 'Ź')
    });
    if !has_at_least_one_letter {
        return false;
    }

    // 2. Reject if non-allowed characters (Chinese, Cyrillic, OCR garbage symbols) exceed 10%
    let total_chars = trimmed.chars().count();
    let invalid_chars = trimmed.chars().filter(|&c| !is_allowed_char(c)).count();
    if invalid_chars > 0 && (invalid_chars as f32 / total_chars as f32) > 0.10 {
        return false;
    }

    // 3. Reject repetitive single character spam (e.g. "aaaaa", "|||||", ".....")
    let clean_sanitized = sanitize_text(trimmed);
    let clean_trimmed = clean_sanitized.trim();
    if clean_trimmed.len() > 1 {
        let first_char = clean_trimmed.chars().next().unwrap();
        if clean_trimmed.chars().all(|c| c == first_char) {
            return false;
        }
    }

    true
}
