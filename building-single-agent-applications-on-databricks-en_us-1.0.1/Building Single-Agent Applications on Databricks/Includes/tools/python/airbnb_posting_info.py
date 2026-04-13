def airbnb_posting_info(id: int) -> str:
    """
    Fetches Airbnb posting information as formatted text.

    Args:
        id (int): Airbnb listing ID (e.g., 958)

    Returns:
        str: Formatted listing information (description, reviews, and rating) or error message
    """
    import re
    import requests

    api_url = f"https://www.airbnb.com/rooms/{id}"

    try:
        response = requests.get(api_url, timeout=10)

        if response.status_code == 200:
            html = response.text

            desc = re.search(r'"metaDescription":"([^"]+)"', html)
            if desc:
                description = desc.group(1).replace('\\n', ' ')
                parts = description.split(' · ')
                description = ' · '.join(parts[2:]) if len(parts) > 2 else description
            else:
                description = "Description not found"

            reviews = re.search(r'"reviewCount":(\d+)', html)
            rating = re.search(r'"starRating":([\d.]+)', html)

            reviews = reviews.group(1) if reviews else "N/A"
            rating = rating.group(1) if rating else "N/A"

            return f"Description: {description}\n\nReviews: {reviews}\nRating: {rating} stars"
        else:
            return f"Request failed with status code: {response.status_code}"

    except Exception as e:
        return f"Request error: {str(e)}"
