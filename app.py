# import library

# basic library
import gdown
import os

import re
import warnings
import joblib
import pdfplumber

import numpy as np
import pandas as pd

# typing
from typing import (
    List,
    Dict,
    Optional
)

# keras
import tensorflow as tf

from tensorflow.keras.models import load_model

# sklearn
from sklearn.metrics.pairwise import (
    cosine_similarity
)

# scipy
from scipy.sparse import (
    load_npz
)

# nlp
from sentence_transformers import (
    SentenceTransformer
)

# generative ai
import google.generativeai as genai

# fastapi
from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.responses import ORJSONResponse

from fastapi.middleware.gzip import GZipMiddleware

from pydantic import BaseModel

# warning
warnings.filterwarnings('ignore')

# gemini config
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

genai.configure(
    api_key=GEMINI_API_KEY
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash-lite"
)

# fastapi app
app = FastAPI(
    title="Hirings AI Service",
    default_response_class=ORJSONResponse
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

# clean text
def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r'<.*?>',
        ' ',
        text
    )

    text = re.sub(
        r'http\S+|www\S+',
        ' ',
        text
    )

    text = re.sub(
        r'[^a-zA-Z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()

# download files
if not os.path.exists(
    'final_jobs_data.csv'
):

    gdown.download(
        'https://drive.google.com/uc?id=1huZCfGSnq9SOGYb-zM9_ATy_wLxYq4dH',
        'final_jobs_data.csv',
        quiet=False
    )

if not os.path.exists(
    'job_embeddings.npy'
):

    gdown.download(
        'https://drive.google.com/uc?id=1ueagG984DBYqbrhLtkOMHe-fisy8O-0N',
        'job_embeddings.npy',
        quiet=False
    )

if not os.path.exists(
    'tfidf_matrix.npz'
):

    gdown.download(
        'https://drive.google.com/uc?id=1ADxNww9brEmaasXU7pkFjVhFXhcQ_vhw',
        'tfidf_matrix.npz',
        quiet=False
    )
if not os.path.exists('forecast_scaler.pkl'):
    import base64
    scaler_b64 = "gASVXAEAAAAAAACMG3NrbGVhcm4ucHJlcHJvY2Vzc2luZy5fZGF0YZSMDE1pbk1heFNjYWxlcpSTlCmBlH2UKIwNZmVhdHVyZV9yYW5nZZRLAEsBhpSMBGNvcHmUiIwEY2xpcJSJjA5uX2ZlYXR1cmVzX2luX5RLAYwPbl9zYW1wbGVzX3NlZW5flEsFjAZzY2FsZV+UjBNqb2JsaWIubnVtcHlfcGlja2xllIwRTnVtcHlBcnJheVdyYXBwZXKUk5QpgZR9lCiMCHN1YmNsYXNzlIwFbnVtcHmUjAduZGFycmF5lJOUjAVzaGFwZZRLAYWUjAVvcmRlcpSMAUOUjAVkdHlwZZRoEowFZHR5cGWUk5SMAmY4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGKMCmFsbG93X21tYXCUiIwbbnVtcHlfYXJyYXlfYWxpZ25tZW50X2J5dGVzlEsQdWII//////////8RERERERGxP5UqAAAAAAAAAIwEbWluX5RoDimBlH2UKGgRaBRoFUsBhZRoF2gYaBloHmghiGgiSxB1YgT/////q6qqqqqqEsCVLwAAAAAAAACMCWRhdGFfbWluX5RoDimBlH2UKGgRaBRoFUsBhZRoF2gYaBloHmghiGgiSxB1Yg////////////////////8AAAAAAIBRQJUvAAAAAAAAAIwJZGF0YV9tYXhflGgOKYGUfZQoaBFoFGgVSwGFlGgXaBhoGWgeaCGIaCJLEHViD////////////////////wAAAAAAQFVAlTEAAAAAAAAAjAtkYXRhX3JhbmdlX5RoDimBlH2UKGgRaBRoFUsBhZRoF2gYaBloHmghiGgiSxB1Yg3/////////////////AAAAAAAALkCVHgAAAAAAAACMEF9za2xlYXJuX3ZlcnNpb26UjAUxLjYuMZR1Yi4="
    with open('forecast_scaler.pkl', 'wb') as f:
        f.write(base64.b64decode(scaler_b64))


# load dataset
df = pd.read_csv(
    'final_jobs_data.csv'
)

salary_df = pd.read_csv(
    'job_salary_mean.csv'
)

# fill missing values
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')

# apply cleaning
df['clean_title'] = (
    df['title']
    .apply(clean_text)
)

df['clean_description'] = (
    df['description']
    .apply(clean_text)
)


# job display data
job_columns = [
    "title",
    "location",
    "formatted_experience_level",
    "min_salary",
    "max_salary",
    "description",
    "job_posting_url"  
]

jobs_display_df = df[job_columns].copy()

jobs_display_df = jobs_display_df.fillna(
    "Not Specified"
)

jobs_display_df["description"] = (
    jobs_display_df["description"]
    .astype(str)
    .str.split(".")
    .str[0]
    + "."
)

jobs_display_data = jobs_display_df.to_dict(
    orient="records"
)

# combine text feature
df['combined_text'] = (

    # prioritize title relevance
    df['clean_title'].astype(str) + ' ' +
    df['clean_title'].astype(str) + ' ' +
    df['clean_title'].astype(str) + ' ' +

    df['clean_description'].astype(str)
)

# precompute lowercase
df['combined_text_lower'] = (
    df['combined_text']
    .astype(str)
    .str.lower()
)

df['title_lower'] = (
    df['title']
    .astype(str)
    .str.lower()
)

# faster numpy access
combined_texts = df[
    'combined_text_lower'
].values

title_texts = df[
    'title_lower'
].values

# load model
tfidf_matrix = load_npz(
    'tfidf_matrix.npz'
)

tfidf_vectorizer = joblib.load(
    'tfidf_vectorizer.pkl'
)

job_embeddings = np.load(
    'job_embeddings.npy'
)

job_embeddings = np.ascontiguousarray(
    job_embeddings
)

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

forecast_model = None
forecast_scaler = None

try:
    import h5py

    inputs = tf.keras.Input(shape=(3, 1), name='input_layer_1')
    x = tf.keras.layers.LSTM(64, name='lstm_1')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', name='dense_3')(x)
    forecast_model = tf.keras.Model(inputs, outputs, name='functional_1')

    with h5py.File('skill_forecast_model.h5', 'r') as f:
        weights = f['model_weights']

        # debug lstm_cell keys
        print("LSTM_CELL KEYS:", list(weights['lstm_1']['lstm_1']['lstm_cell'].keys()))

        lstm_cell = weights['lstm_1']['lstm_1']['lstm_cell']
        lstm_keys = list(lstm_cell.keys())
        print("LSTM_CELL ALL KEYS:", lstm_keys)

        lstm_layer = forecast_model.get_layer('lstm_1')
        lstm_layer.set_weights([
            lstm_cell['kernel'][()],
            lstm_cell['recurrent_kernel'][()],
            lstm_cell['bias'][()]
        ])

        dense2_cell = weights['dense_2']['dense_2']
        dense2_keys = list(dense2_cell.keys())
        print("DENSE2 KEYS:", dense2_keys)

        dense2_layer = forecast_model.get_layer('dense_2')
        dense2_layer.set_weights([
            dense2_cell['kernel'][()],
            dense2_cell['bias'][()]
        ])

        dense3_cell = weights['dense_3']['dense_3']
        dense3_keys = list(dense3_cell.keys())
        print("DENSE3 KEYS:", dense3_keys)

        dense3_layer = forecast_model.get_layer('dense_3')
        dense3_layer.set_weights([
            dense3_cell['kernel'][()],
            dense3_cell['bias'][()]
        ])

    forecast_scaler = joblib.load('forecast_scaler.pkl')
    print("Forecast model loaded successfully")

except Exception as e:
    print(f"Forecast model failed to load: {type(e).__name__}: {e}")

# skill mapping
skill_mapping = {

    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'js': 'javascript',
    'tf': 'tensorflow'

}

# normalize skills
def normalize_skills(text):

    text = str(text).lower()

    for short, full in skill_mapping.items():

        text = text.replace(
            short,
            full
        )

    return text


# salary recommendation
def get_salary_recommendation(role):

    filtered = salary_df[

        salary_df[
            'Judul Pekerjaan'
        ]

        .str.lower()

        .str.contains(
            role.lower(),
            na=False
        )
    ]

    if filtered.empty:

        return {

            'industry': 'Unknown',
            'salary_min': 0,
            'salary_max': 0,
            'salary_average': 0,
            'company_example': 'Unknown',
            'location_example': 'Unknown'
        }

    salary_avg = filtered[
        'Gaji_Rata2'
    ].mean()

    salary_min = filtered[
        'Gaji_Rata2'
    ].min()

    salary_max = filtered[
        'Gaji_Rata2'
    ].max()

    company_sample = filtered[
        'Perusahaan'
    ].iloc[0]

    location_sample = filtered[
        'Lokasi'
    ].iloc[0]

    return {

        'industry': role,

        'salary_min': round(
            salary_min,
            2
        ),

        'salary_max': round(
            salary_max,
            2
        ),

        'salary_average': round(
            salary_avg,
            2
        ),

        'company_example': company_sample,

        'location_example': location_sample
    }


# roadmap generation
def get_learning_roadmap(role):

    role = role.lower()

    roadmap_dict = {

        'data scientist': [

            'Python',
            'Machine Learning',
            'Deep Learning',
            'SQL',
            'TensorFlow'
        ],

        'frontend developer': [

            'HTML',
            'CSS',
            'JavaScript',
            'React',
            'API Integration'
        ],

        'backend developer': [

            'Node.js',
            'Database',
            'API Development',
            'Authentication',
            'Deployment'
        ]
    }

    for key in roadmap_dict:

        if key in role:

            return roadmap_dict[key]

    return [

        'Communication',
        'Problem Solving',
        'Portfolio Building'
    ]


# cv parsing
def parse_cv(file_path):

    text = ""

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:

                text += extracted + " "

    skills_list = [

        "Python",
        "SQL",
        "Machine Learning",
        "TensorFlow",
        "React",
        "Node.js",
        "MongoDB"
    ]

    found_skills = []

    for skill in skills_list:

        if skill.lower() in text.lower():

            found_skills.append(skill)

    return {

        "skills": found_skills,
        "raw_text": text[:500]
    }


# recommendation function
def recommend_jobs(user_input):

    user_input = normalize_skills(
        user_input
    )

    user_embedding = embedding_model.encode(
        [user_input]
    )

    embedding_similarity = cosine_similarity(
        user_embedding,
        job_embeddings
    )[0]

    user_tfidf = tfidf_vectorizer.transform(
        [user_input]
    )

    tfidf_similarity = cosine_similarity(
        user_tfidf,
        tfidf_matrix
    )[0]

    final_similarity = (

        0.45 * embedding_similarity +
        0.55 * tfidf_similarity
    )

    user_skills = user_input.lower().split()

    # limit skill count
    user_skills = user_skills[:10]

    for idx, text in enumerate(
        combined_texts
    ):

        title = title_texts[idx]

        boost = 0

        for skill in user_skills:

            if skill in text:

                if skill in [

                    'laravel',
                    'django',
                    'flask',
                    'react',
                    'flutter',
                    'tensorflow',
                    'pytorch',
                    'docker',
                    'kubernetes'

                ]:

                    boost += 0.25

                elif skill in [

                    'python',
                    'java',
                    'javascript',
                    'php',
                    'golang'

                ]:

                    boost += 0.15

                else:

                    boost += 0.05

            if skill in title:

                boost += 0.80

        final_similarity[idx] += boost

    sorted_indices = np.argsort(
        final_similarity
    )[::-1]

    top_indices = []

    for i in sorted_indices:

        if final_similarity[i] < 0.35:
            continue

        job_text = str(
            df.iloc[i]['combined_text']
        ).lower()

        job_title = str(
            df.iloc[i]['title']
        ).lower()

        matched_skills = []

        for skill in user_skills:

            if (
                skill in job_text or
                skill in job_title
            ):

                matched_skills.append(
                    skill
                )

        if len(matched_skills) >= max(
            2,
            len(user_skills) // 3
        ):

            top_indices.append(i)

        if len(top_indices) >= 10:
            break

    recommendations = df.iloc[
        top_indices
    ][[

        'job_id',
        'company_id',
        'title',
        'location',
        'formatted_experience_level',
        'min_salary',
        'max_salary',
        'avg_salary',
        'salary_range',
        'remote_allowed',
        'job_posting_url'
    ]].copy()

    recommendations[
        'similarity_score'
    ] = final_similarity[top_indices]
        
    roadmaps = []

    for title in recommendations['title']:

        roadmap = get_learning_roadmap(
            title
        )

        roadmaps.append(
            roadmap
        )

    recommendations[
        'roadmap'
    ] = roadmaps

    recommendations = recommendations.sort_values(
        by='similarity_score',
        ascending=False
    )

    recommendations = recommendations.fillna(0)

    return recommendations.to_dict(
        orient='records'
    )


# forecasting function
def forecast_skill_trend(trend):

    if forecast_model is None or forecast_scaler is None:

        return 0.0

    input_scaled = forecast_scaler.transform(
        np.array(trend).reshape(-1, 1)
    )

    input_scaled = input_scaled.reshape(1, 3, 1)

    prediction = forecast_model.predict(input_scaled)

    prediction = forecast_scaler.inverse_transform(prediction)

    return float(prediction[0][0])

# prompt builder
def build_career_prompt(

    user_message,
    context

):

    return f"""

    Kamu adalah AI Career Assistant
    untuk platform Hirings.

    USER MESSAGE:
    {user_message}

    USER SKILLS:
    {context.get('skills', [])}

    CV:
    {context.get('cv_raw_text', '')}

    SKILL GAP:
    {context.get('skill_gap', {})}

    CAREER RECOMMENDATIONS:
    {context.get('career_recommendations', [])}

    Tugas kamu:
    - Jawab pertanyaan user
    - Berikan career advice
    - Jelaskan skill gap
    - Berikan roadmap belajar
    - Berikan saran pengembangan skill

    Gunakan bahasa Indonesia.
    Maksimal 200 kata.
    """

# generate llm response
def generate_career_response(

    user_message,
    context

):

    if (

        len(
            context.get(
                "skills",
                []
            )
        ) == 0

        and

        context.get(
            "cv_raw_text",
            ""
        ).strip() == ""
    ):

        return {

            "success": False,

            "reply": (

                "Belum bisa menganalisis "
                "profil Anda karena "
                "profil atau CV belum lengkap."
            ),

            "recommendations": [],
            "next_steps": []
        }

    prompt = build_career_prompt(
        user_message,
        context
    )

    response = gemini_model.generate_content(
        prompt
    )

    reply_text = response.text

    return {

        "success": True,

        "reply": reply_text,

        "recommendations": context.get(
            "career_recommendations",
            []
        ),

        "next_steps": context.get(
            "skill_gap",
            {}
        ).get(
            "missing_skills",
            []
        )
    }


# request schema
class RecommendationRequest(
    BaseModel
):

    skills: str


class ForecastRequest(
    BaseModel
):

    trend: List[float]


class CareerChatRequest(
    BaseModel
):

    user_message: str
    context: Dict


# api endpoint
@app.get("/")
def home():

    return {

        "message": "Hirings AI Service Running"
    }


@app.post("/recommend")
def recommend_api(

    request: RecommendationRequest
):

    result = recommend_jobs(
        request.skills
    )

    return result


@app.post("/forecast")
def forecast_api(

    request: ForecastRequest
):

    prediction = forecast_skill_trend(
        request.trend
    )

    return {

        "prediction": prediction
    }


@app.post("/career-suggestion")
def career_suggestion_api(

    request: RecommendationRequest
):

    recommendations = recommend_jobs(
        request.skills
    )

    return {

        "career_suggestion":

        f"Based on your skills in "
        f"{request.skills}, "
        f"you are suitable for "
        f"technology-related careers."
    }

@app.post('/parse-cv')
async def parse_cv_api(

    file: UploadFile = File(...)

):

    temp_path = f'temp_{file.filename}'
    with open(temp_path, 'wb') as f:
        content = await file.read()
        f.write(content)
        
    try:
        parsed_data = parse_cv(temp_path)
        return {
            'success': True,
            'data': parsed_data
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.post("/ai-career-chat")
def ai_career_chat(

    request: CareerChatRequest
):

    result = generate_career_response(

        request.user_message,
        request.context
    )

    return result
    

@app.get("/all-job")
def get_all_jobs():

    return {

        "total_jobs": len(jobs_display_data),
        "jobs": jobs_display_data
    }

# run app
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860
    )