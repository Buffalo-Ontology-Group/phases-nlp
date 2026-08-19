"""
Check whether a psychological term has an entry in the
APA Dictionary of Psychology.

This module does not extract or reproduce the APA definition.
It only returns the official APA Dictionary URL when an entry
can be verified.
"""

import re
from urllib.parse import unquote

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


APA_BASE_URL = "https://dictionary.apa.org"


APA_NOT_FOUND_PATTERNS = [
    r"not\s+in\s+the\s+dictionary\s+of\s+psychology",
    r"not\s+in\s+the\s+apa\s+dictionary",
    r"no\s+dictionary\s+entry",
    r"entry\s+was\s+not\s+found",
    r"term\s+was\s+not\s+found",
    r"page\s+not\s+found",
    r"please\s+report\s+to\s+apa",
    r"sorry.{0,200}not\s+in\s+the\s+dictionary",
]


GENERIC_PAGE_TEXT = [
    "apa dictionary of psychology",
    "a trusted reference in the field of psychology",
    "offering more than 25,000",
    "more than 25,000 authoritative entries",
]


def normalize_term(term: str) -> str:
    """
    Normalize a term for text comparison.

    Examples:
        "Social Isolation" -> "social isolation"
        "self-transcendence" -> "self transcendence"
    """

    term = unquote(term)
    term = term.lower().strip()
    term = re.sub(r"[-–—_]+", " ", term)
    term = re.sub(r"[^a-z0-9\s]", "", term)
    term = re.sub(r"\s+", " ", term)

    return term.strip()


def slugify_term(term: str) -> str:
    """
    Convert a term into an APA Dictionary URL slug.

    Examples:
        "loneliness" -> "loneliness"
        "social isolation" -> "social-isolation"
        "self-transcendence" -> "self-transcendence"
    """

    normalized = normalize_term(term)
    return normalized.replace(" ", "-")


def normalize_page_text(text: str) -> str:
    """
    Normalize rendered page text for comparison.
    """

    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[-–—_]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_missing_entry_message(page_text: str) -> bool:
    """
    Check whether the rendered page explicitly indicates
    that the requested term does not have an APA entry.
    """

    if not page_text:
        return False

    return any(
        re.search(
            pattern,
            page_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        for pattern in APA_NOT_FOUND_PATTERNS
    )


def is_generic_apa_page(page_text: str) -> bool:
    """
    Determine whether the rendered page contains only generic
    APA Dictionary material rather than a specific term entry.
    """

    normalized_text = normalize_page_text(page_text)

    return any(
        generic_text in normalized_text
        for generic_text in GENERIC_PAGE_TEXT
    )


def term_matches_text(term: str, candidate_text: str) -> bool:
    """
    Compare the requested term with a possible entry heading,
    page title, or URL slug.
    """

    normalized_term = normalize_term(term)
    normalized_candidate = normalize_term(candidate_text)

    if not normalized_term or not normalized_candidate:
        return False

    if normalized_term == normalized_candidate:
        return True

    # Some APA headings may contain part-of-speech labels,
    # abbreviations, or punctuation after the entry term.
    if normalized_candidate.startswith(
        f"{normalized_term} "
    ):
        return True

    return False


def find_entry_heading(page, term: str) -> str | None:
    """
    Search rendered headings for one matching the requested term.
    """

    selectors = [
        "h1",
        "h2",
        "h3",
        "[role='heading']",
        "[class*='title']",
        "[class*='term']",
        "[class*='entry'] h1",
        "[class*='entry'] h2",
        "[class*='entry'] h3",
    ]

    for selector in selectors:
        try:
            elements = page.locator(selector)
            count = min(elements.count(), 50)

            for index in range(count):
                text = elements.nth(index).inner_text(
                    timeout=2_000
                ).strip()

                if term_matches_text(term, text):
                    return text

        except PlaywrightError:
            continue

    return None


def get_apa_dictionary_entry(
    term: str,
    *,
    headless: bool = True,
    timeout_ms: int = 30_000,
) -> dict:
    """
    Check whether a term has an APA Dictionary entry.

    This function does not extract the APA definition.

    Returns a dictionary with one of these statuses:

    - found:
        A matching APA Dictionary entry was verified.

    - not_found:
        The rendered APA page explicitly indicated that the
        term does not have an entry.

    - unavailable:
        The page loaded, but the script could not reliably
        confirm whether an entry exists.

    - request_error:
        The APA website could not be accessed.

    Returned dictionary example:

        {
            "term": "loneliness",
            "source": "APA Dictionary of Psychology",
            "url": "https://dictionary.apa.org/loneliness",
            "status": "found",
            "entry_heading": "loneliness",
        }
    """

    original_term = term.strip()

    if not original_term:
        raise ValueError(
            "The APA Dictionary term cannot be empty."
        )

    slug = slugify_term(original_term)
    requested_url = f"{APA_BASE_URL}/{slug}"

    result = {
        "term": original_term,
        "source": "APA Dictionary of Psychology",
        "url": requested_url,
        "status": "unavailable",
    }

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=headless,
            )

            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 "
                    "(Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                locale="en-US",
            )

            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            response = page.goto(
                requested_url,
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )

            if response is None:
                browser.close()

                return {
                    **result,
                    "status": "request_error",
                    "error": (
                        "The APA Dictionary returned no "
                        "HTTP response."
                    ),
                }

            status_code = response.status

            if status_code == 404:
                browser.close()

                return {
                    **result,
                    "status": "not_found",
                    "http_status": status_code,
                }

            if status_code >= 400:
                browser.close()

                return {
                    **result,
                    "status": "request_error",
                    "http_status": status_code,
                    "error": (
                        "The APA Dictionary returned HTTP "
                        f"status {status_code}."
                    ),
                }

            # Give client-rendered content time to appear.
            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=10_000,
                )
            except PlaywrightTimeoutError:
                # The page may keep background connections
                # open. Continue with the rendered content
                # already available.
                pass

            try:
                page.wait_for_timeout(2_000)
            except PlaywrightError:
                pass

            final_url = page.url
            result["url"] = final_url
            result["http_status"] = status_code

            try:
                page_title = page.title().strip()
            except PlaywrightError:
                page_title = ""

            try:
                body_text = page.locator(
                    "body"
                ).inner_text(
                    timeout=10_000
                )
            except PlaywrightError:
                body_text = ""

            normalized_body = normalize_page_text(
                body_text
            )

            # A clear missing-entry message is the strongest
            # evidence that the term is absent.
            if contains_missing_entry_message(
                normalized_body
            ):
                browser.close()

                return {
                    **result,
                    "status": "not_found",
                    "page_title": page_title,
                }

            # Look for a rendered heading that exactly matches
            # the requested term.
            entry_heading = find_entry_heading(
                page,
                original_term,
            )

            if entry_heading:
                browser.close()

                return {
                    **result,
                    "status": "found",
                    "entry_heading": entry_heading,
                    "page_title": page_title,
                }

            # Check whether the final URL still contains the
            # expected APA term slug.
            final_slug = final_url.rstrip("/").split("/")[-1]
            final_slug = final_slug.split("?")[0]
            final_slug = normalize_term(final_slug)

            expected_term = normalize_term(original_term)

            url_matches = (
                final_slug == expected_term
            )

            # Existing APA entry pages commonly include the
            # term in the rendered page body.
            body_contains_term = bool(
                re.search(
                    rf"\b{re.escape(expected_term)}\b",
                    normalized_body,
                )
            )

            if url_matches and body_contains_term:
                browser.close()

                return {
                    **result,
                    "status": "found",
                    "page_title": page_title,
                }

            # The generic application page does not give enough
            # evidence to claim that the entry is absent.
            if is_generic_apa_page(normalized_body):
                browser.close()

                return {
                    **result,
                    "status": "unavailable",
                    "page_title": page_title,
                    "error": (
                        "The APA Dictionary page loaded, "
                        "but a term-specific entry could not "
                        "be verified automatically."
                    ),
                }

            browser.close()

            return {
                **result,
                "status": "unavailable",
                "page_title": page_title,
                "error": (
                    "The APA Dictionary page loaded, but the "
                    "entry status could not be determined "
                    "reliably."
                ),
            }

    except PlaywrightTimeoutError as error:
        return {
            **result,
            "status": "request_error",
            "error": (
                "The APA Dictionary request timed out: "
                f"{error}"
            ),
        }

    except PlaywrightError as error:
        return {
            **result,
            "status": "request_error",
            "error": str(error),
        }


# Compatibility alias.
#
# This allows existing main.py code that currently imports
# get_apa_dictionary_definition() to continue working.
# The function now checks only for an entry and URL; it does
# not retrieve a definition.
def get_apa_dictionary_definition(
    term: str,
) -> dict:
    """
    Compatibility wrapper for older code.

    Despite the old function name, no definition is extracted.
    """

    return get_apa_dictionary_entry(term)


if __name__ == "__main__":
    test_terms = [
        "loneliness",
        "gerotranscendence",
    ]

    for test_term in test_terms:
        entry = get_apa_dictionary_entry(
            test_term,
            headless=True,
        )

        print("=" * 60)
        print("Term:", entry["term"])
        print("Status:", entry["status"])
        print("URL:", entry["url"])

        if entry.get("entry_heading"):
            print(
                "Entry heading:",
                entry["entry_heading"],
            )

        if entry.get("error"):
            print("Message:", entry["error"])