from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st


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
        "us-zip-code-latitude-and-longitude.csv",
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
        "Zip": best_zips.astype(str)
    })

    df["Zip"] = df["Zip"].str.zfill(5)

    # Merge ZIP recommendations with city/state data
    merged = pd.merge(
        df,
        zip_location,
        on="Zip",
        how="inner"
    )

    merged = merged[["Zip", "City", "State"]]

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

    # Return requested number of recommendations
    return merged.head(int(num)), None


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

            st.dataframe(
                recommendations,
                hide_index=True,
                use_container_width=True
            )
