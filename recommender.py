from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import html
import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from urllib.parse import quote_plus


@st.cache_data
def load_data():
    # Load recommendation data
    final = pd.read_csv("final_data.csv")

    # Keep ZIP codes as strings and preserve leading zeros
    final["index"] = final["index"].astype(str).str.zfill(5)

    # Remove unnecessary columns if they exist
    final.drop(
        columns=["Unnamed: 0", "level_0"],
        inplace=True,
        errors="ignore"
    )

    final.set_index("index", inplace=True)

    # Load ZIP location data
    zip_location = pd.read_csv(
        "us-zip-code-latitude-and-longitude (1).csv",
        sep=";",
        dtype={"Zip": str}
    )

    # Preserve leading zeros in ZIP codes
    zip_location["Zip"] = zip_location["Zip"].str.zfill(5)

    return final, zip_location


@st.cache_data
def prepare_scaled_data(final):
    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(final)

    scaled = pd.DataFrame(
        scaled_values,
        index=final.index,
        columns=final.columns
    )

    return scaled


def add_result_links(recommendations):
    """Add ZIP-specific data and town-information links to each result."""
    recommendations = recommendations.copy()

    recommendations["ZIP Data"] = recommendations["Zip"].apply(
        lambda zip_code: (
            "https://censusreporter.org/profiles/"
            f"86000US{zip_code}-{zip_code}/"
        )
    )

    recommendations["Wikipedia"] = recommendations.apply(
        lambda row: (
            "https://en.wikipedia.org/w/index.php?search="
            + quote_plus(f"{row['City']}, {row['State']}")
        ),
        axis=1
    )

    recommendations["Map"] = recommendations["Zip"].apply(
        lambda zip_code: (
            "https://www.google.com/maps/search/?api=1&query="
            + quote_plus(str(zip_code))
        )
    )

    return recommendations


def display_recommendation_map(recommendations):
    """Display numbered, clickable recommendations on an interactive U.S. map."""
    map_rows = recommendations.dropna(
        subset=["Latitude", "Longitude"]
    ).copy()

    if map_rows.empty:
        st.info("Map coordinates are not available for these recommendations.")
        return

    markers = []

    for _, row in map_rows.iterrows():
        markers.append({
            "rank": int(row["Rank"]),
            "zip": html.escape(str(row["Zip"])),
            "city": html.escape(str(row["City"])),
            "state": html.escape(str(row["State"])),
            "similarity": float(row["Similarity %"]),
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
            "zip_data": row["ZIP Data"],
            "wikipedia": row["Wikipedia"],
            "map": row["Map"]
        })

    marker_json = json.dumps(markers)

    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet"
              href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <style>
            html, body, #map {{
                height: 100%;
                margin: 0;
                font-family: Arial, sans-serif;
            }}
            .rank-marker {{
                width: 32px;
                height: 32px;
                border-radius: 50%;
                background: #d62728;
                border: 2px solid white;
                color: white;
                font-weight: bold;
                line-height: 28px;
                text-align: center;
                box-shadow: 0 1px 5px rgba(0, 0, 0, 0.55);
            }}
            .popup-title {{
                font-size: 15px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .popup-links {{ margin-top: 8px; }}
            .popup-links a {{ margin-right: 10px; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            const recommendations = {marker_json};
            const map = L.map('map');

            L.tileLayer(
                'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
                {{
                    maxZoom: 19,
                    attribution: '&copy; OpenStreetMap contributors'
                }}
            ).addTo(map);

            const bounds = [];

            recommendations.forEach((place) => {{
                const icon = L.divIcon({{
                    className: '',
                    html: `<div class="rank-marker">${{place.rank}}</div>`,
                    iconSize: [32, 32],
                    iconAnchor: [16, 16]
                }});

                const popup = `
                    <div class="popup-title">
                        ${{place.rank}}. ${{place.city}}, ${{place.state}}
                    </div>
                    <div>ZIP ${{place.zip}}</div>
                    <div>${{place.similarity.toFixed(1)}}% similar</div>
                    <div class="popup-links">
                        <a href="${{place.zip_data}}" target="_blank">ZIP data</a>
                        <a href="${{place.wikipedia}}" target="_blank">Town info</a>
                        <a href="${{place.map}}" target="_blank">Google Maps</a>
                    </div>`;

                L.marker(
                    [place.latitude, place.longitude],
                    {{icon: icon}}
                ).addTo(map).bindPopup(popup);

                bounds.push([place.latitude, place.longitude]);
            }});

            if (bounds.length === 1) {{
                map.setView(bounds[0], 9);
            }} else {{
                map.fitBounds(bounds, {{padding: [35, 35], maxZoom: 9}});
            }}
        </script>
    </body>
    </html>
    """

    components.html(map_html, height=560)


def cosine_recommend_zip(zip_input, num, city, state):
    # Clean user input
    zip_input = str(zip_input).strip().zfill(5)
    city = city.strip()
    state = state.strip().upper()

    # Load data
    final, zip_location = load_data()

    # Scale recommendation variables
    scaled = prepare_scaled_data(final)

    # Check whether ZIP exists
    if zip_input not in scaled.index:
        return None, "Please enter a valid ZIP code."

    # Find row corresponding to input ZIP
    zip_index = scaled.index.get_loc(zip_input)

    # Compare input ZIP only against all other ZIP codes
    cosine_scores = cosine_similarity(
        scaled.iloc[[zip_index]],
        scaled
    )[0]

    # Sort from most similar to least similar
    zip_indices = cosine_scores.argsort()[::-1]

    # Get ZIP codes in similarity order
    best_zips = scaled.index[zip_indices]

    # IMPORTANT: Keep ZIP codes as strings
    df = pd.DataFrame({
        "Zip": best_zips.astype(str),
        "Similarity %": cosine_scores[zip_indices]
    })

    df["Zip"] = df["Zip"].str.zfill(5)

    # Merge ZIP recommendations with city/state data
    merged = pd.merge(
        df,
        zip_location,
        on="Zip",
        how="inner"
    )

    merged = merged[
        ["Zip", "City", "State", "Latitude", "Longitude", "Similarity %"]
    ]

    # Remove the user's own ZIP code
    merged = merged[merged["Zip"] != zip_input]

    # Filter by state if provided
    if state:
        merged = merged[
            merged["State"].astype(str).str.upper() == state
        ]

    # Filter by city if provided
    if city:
        if not state:
            return None, "You must enter a state when entering a city."

        merged = merged[
            merged["City"].astype(str).str.lower()
            == city.lower()
        ]

    # Reset row numbers
    merged = merged.reset_index(drop=True)

    # No recommendations found
    if merged.empty:
        return None, "No matching recommendations found."

    # Return requested number of recommendations with useful links
    recommendations = merged.head(int(num)).copy()
    recommendations["Similarity %"] = (
        recommendations["Similarity %"].clip(lower=0, upper=1) * 100
    ).round(1)
    recommendations.insert(
        0,
        "Rank",
        range(1, len(recommendations) + 1)
    )
    recommendations = add_result_links(recommendations)

    return recommendations, None


# -----------------------------
# STREAMLIT PAGE
# -----------------------------

st.set_page_config(
    page_title="Relocation Recommender",
    page_icon="🏠",
    layout="centered"
)

st.title("Relocation Recommender System")

st.write(
    "Enter your current ZIP code to find similar places to live."
)

zip_input = st.text_input(
    "ZIP Code",
    max_chars=5,
    placeholder="Example: 78701"
)

num = st.number_input(
    "Number of recommendations",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)

state = st.text_input(
    "State abbreviation (optional)",
    max_chars=2,
    placeholder="Example: TX"
)

city = st.text_input(
    "City (optional)",
    placeholder="Example: Austin"
)

if st.button("Find Recommendations", type="primary"):

    zip_input = zip_input.strip()

    if not zip_input:
        st.error("Please enter a ZIP code.")

    elif not zip_input.isdigit():
        st.error("ZIP code must contain numbers only.")

    elif len(zip_input) > 5:
        st.error("Please enter a valid 5-digit ZIP code.")

    else:
        # This allows someone to type 2108 and treats it as 02108
        zip_input = zip_input.zfill(5)

        with st.spinner("Finding similar ZIP codes..."):

            recommendations, error = cosine_recommend_zip(
                zip_input,
                num,
                city,
                state
            )

        if error:
            st.error(error)

        else:
            st.subheader("Recommended ZIP Codes")

            display_recommendation_map(recommendations)

            st.caption(
                "Similarity is the cosine-similarity score across the "
                "relocation variables, expressed on a 0–100 scale."
            )

            st.dataframe(
                recommendations,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("#", format="%d"),
                    "Zip": st.column_config.TextColumn("ZIP Code"),
                    "Similarity %": st.column_config.ProgressColumn(
                        "Similarity",
                        help="Cosine similarity across the relocation variables",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "Latitude": None,
                    "Longitude": None,
                    "ZIP Data": st.column_config.LinkColumn(
                        "ZIP Data",
                        help="Demographic and housing data from Census Reporter",
                        display_text="View ZIP data"
                    ),
                    "Wikipedia": st.column_config.LinkColumn(
                        "Town Info",
                        help="Search Wikipedia for the city or town",
                        display_text="View town"
                    ),
                    "Map": st.column_config.LinkColumn(
                        "Map",
                        help="Open this ZIP code in Google Maps",
                        display_text="Open map"
                    )
                }
            )
