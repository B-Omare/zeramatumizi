"""
rag_pipeline.py
RAG-powered NACADA counsellor assistant for ZeraMatumizi.

A NACADA officer or school counsellor asks a question in Swahili
or English and receives a protocol-grounded, stepwise response.

Stack:
- Embeddings: sentence-transformers (multilingual, supports Swahili)
- Vector store: ChromaDB (local, no server needed)
- LLM: Google Gemini (free tier)
"""

import os
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CHROMA_PATH = os.path.join("data", "chroma_db")
REPORTS_PATH = os.path.join("docs", "reports")


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)


def create_knowledge_base() -> list:
    """
    Creates the ZeraMatumizi knowledge base.

    In production this will be loaded from:
    - Kenya National Protocol for Treatment of Substance Use Disorders 2017
    - Kenya Mental Health Policy 2015-2030
    - NACADA Strategic Documents
    - WHO ASSIST Guidelines
    - UNODC Community Treatment Manuals
    """
    documents = [
        {
            "id": "nacada_001",
            "source": "NACADA Protocol 2022",
            "content": """
            When a student is suspected of using substances, the school counsellor
            should: (1) Approach the student privately and non-judgmentally.
            (2) Conduct a brief screening using the WHO ASSIST tool.
            (3) Score the ASSIST: Low risk (0-3) = brief advice;
            Moderate risk (4-26) = brief intervention and monitoring;
            High risk (27+) = referral to specialist treatment.
            (4) Document the interaction in the school health register.
            (5) Involve parents or guardians with student consent where appropriate.
            (6) Never shame or punish the student for substance use.
            """,
            "tags": ["student", "school", "screening", "ASSIST", "counsellor"]
        },
        {
            "id": "nacada_002",
            "source": "NACADA Protocol 2022",
            "content": """
            The referral pathway for substance use disorders in Kenya:
            Level 1: Community Health Worker (CHW) - initial identification
            and brief advice in the community.
            Level 2: Primary Health Care facility - brief intervention,
            ASSIST screening, basic counselling.
            Level 3: County Referral Hospital - specialist outpatient
            treatment, medication-assisted treatment.
            Level 4: National Referral - Mathare Hospital, Nairobi;
            Coast Provincial General Hospital.
            Private facilities: Chiromo Hospital Group, Asumbi Treatment Centre.
            NACADA helpline: 1192 (toll-free, 24 hours).
            """,
            "tags": ["referral", "pathway", "hospital", "CHW", "helpline"]
        },
        {
            "id": "nacada_003",
            "source": "NACADA Protocol 2022",
            "content": """
            Chang'aa and illicit brew poisoning response protocol:
            If a community member presents with suspected poisoning from
            illicit alcohol (chang'aa, busaa, or other illicit brews):
            (1) Call emergency services immediately - 999 or 0800 723 253.
            (2) Do not induce vomiting.
            (3) Keep the person awake and breathing.
            (4) Transport to nearest health facility immediately.
            (5) Report the incident to NACADA via 1192 and to the
            local police for source identification.
            Methanol poisoning from illicit brew requires urgent
            fomepizole or ethanol antidote treatment.
            """,
            "tags": ["poisoning", "changaa", "emergency", "illicit brew", "methanol"]
        },
        {
            "id": "assist_001",
            "source": "WHO ASSIST Guidelines",
            "content": """
            The WHO Alcohol, Smoking and Substance Involvement Screening Test
            (ASSIST) covers tobacco, alcohol, cannabis, cocaine, amphetamines,
            inhalants, sedatives, hallucinogens, opioids, and other drugs.
            It consists of 8 questions covering lifetime use, past 3 months use,
            urge to use, health problems, failure to fulfil role obligations,
            concern from others, and attempts to cut down.
            ASSIST scores guide intervention intensity:
            - Score 0-3 (most substances) or 0-10 (alcohol): No intervention
            - Score 4-26 (most substances) or 11-26 (alcohol): Brief intervention
            - Score 27+ : Referral to specialist treatment
            Brief intervention takes 5-15 minutes and uses motivational
            interviewing techniques.
            """,
            "tags": ["ASSIST", "screening", "score", "intervention", "motivational"]
        },
        {
            "id": "assist_002",
            "source": "WHO ASSIST Guidelines",
            "content": """
            Brief intervention for substance use - FRAMES approach:
            F - Feedback: Share personalised risk information with the client.
            R - Responsibility: Emphasise personal choice and responsibility.
            A - Advice: Give clear advice to reduce or stop use.
            M - Menu: Offer a range of options for change.
            E - Empathy: Use a warm, reflective, non-judgemental style.
            S - Self-efficacy: Reinforce the client's belief in their ability to change.
            This approach is effective for alcohol and cannabis use disorders
            and can be delivered by non-specialist health workers.
            """,
            "tags": ["brief intervention", "FRAMES", "motivational", "counselling"]
        },
        {
            "id": "kmhp_001",
            "source": "Kenya Mental Health Policy 2015-2030",
            "content": """
            Kenya's Mental Health Policy recognises substance use disorders
            as a priority mental health condition. Key policy commitments:
            (1) Integration of mental health and substance use services
            into primary health care by 2030.
            (2) Reduction of treatment gap from 90% to 50% by 2030.
            (3) Training of at least one mental health professional per
            primary health care facility.
            (4) Community-based treatment as the preferred modality over
            institutional treatment.
            (5) Anti-stigma campaigns in all 47 counties.
            County health departments are mandated to include substance
            use disorder treatment in their Annual Work Plans.
            """,
            "tags": ["policy", "mental health", "treatment gap", "county", "primary care"]
        },
        {
            "id": "cannabis_001",
            "source": "NACADA Protocol 2022",
            "content": """
            Cannabis (bhang) use among students - response protocol:
            Cannabis is the most commonly used illicit substance among
            Kenyan youth aged 15-24, with Nairobi, Nyanza, and Coast
            regions having highest prevalence.
            Signs of cannabis use: red eyes, increased appetite,
            slowed reaction time, impaired memory, paranoia.
            For a student found using cannabis:
            (1) Conduct private, non-judgemental conversation.
            (2) Administer ASSIST screening.
            (3) For moderate risk: 3-session brief intervention
            using cognitive behavioural techniques.
            (4) Monitor school attendance and academic performance monthly.
            (5) Refer parents for family counselling if home environment
            is a contributing factor.
            Do not expel or suspend - this increases dropout risk.
            """,
            "tags": ["cannabis", "bhang", "student", "youth", "school"]
        },
        {
            "id": "alcohol_001",
            "source": "NACADA Protocol 2022",
            "content": """
            Alcohol use disorder treatment in Kenya:
            Mild to moderate alcohol use disorder:
            - Brief intervention (FRAMES) at primary care level
            - Motivational enhancement therapy (4 sessions)
            - Self-help groups (AA Kenya, available in Nairobi, Mombasa,
              Kisumu, Nakuru, Eldoret)
            Severe alcohol use disorder:
            - Medical detoxification under supervision (benzodiazepines
              for withdrawal management)
            - Medication-assisted treatment: naltrexone, acamprosate,
              disulfiram (available at county referral hospitals)
            - Residential rehabilitation: 28-90 day programmes
            AUDIT-C screening: 3 questions on frequency, quantity,
            and binge drinking. Score 4+ in men, 3+ in women indicates
            hazardous drinking requiring brief intervention.
            """,
            "tags": ["alcohol", "AUDIT", "detox", "treatment", "medication"]
        },
        {
            "id": "swahili_001",
            "source": "NACADA Swahili Resources",
            "content": """
            Jinsi ya kuzungumza na mwanafunzi anayetumia dawa za kulevya:
            1. Mwite pembeni kwa siri - Call them aside privately
            2. Zungumza kwa upole na heshima - Speak gently and respectfully
            3. Sikiliza zaidi kuliko kuzungumza - Listen more than you speak
            4. Usilaumu wala kumhukumu - Do not blame or judge
            5. Mwulize: Je, unaweza kunieleza zaidi kuhusu hali yako?
            6. Toa taarifa kwa NACADA: 1192 (bure)
            Neno la msaada: Kupona kunawezekana. Msaada unapatikana.
            """,
            "tags": ["swahili", "student", "communication", "counsellor", "school"]
        },
        {
            "id": "swahili_002",
            "source": "NACADA Swahili Resources",
            "content": """
            Dalili za matumizi ya dawa za kulevya (Signs of drug use):
            Pombe (Alcohol): harufu ya pombe, macho mekundu, kutembea
            kwa shida, kuzungumza bila mpangilio.
            Bhang (Cannabis): macho mekundu, kicheko bila sababu,
            kula sana, usingizi mwingi.
            Miraa (Khat): kutolala, kuzungumza sana, msisimko.
            Chang'aa: harufu kali, kuanguka, kukosa fahamu.
            Hatua za kwanza (First steps):
            - Piga simu NACADA: 1192
            - Mpeleke hospitali ya karibu
            - Usimwache peke yake
            """,
            "tags": ["swahili", "signs", "symptoms", "changaa", "bhang", "miraa"]
        },
        {
            "id": "chw_001",
            "source": "UNODC Community Treatment Manual",
            "content": """
            Community Health Worker (CHW) role in substance use disorder:
            CHWs are the first point of contact in Kenya's community health
            strategy. For substance use disorders CHWs should:
            (1) Identify at-risk individuals during household visits using
            the CAGE questionnaire (4 questions on alcohol).
            (2) Provide brief advice (5 minutes) on reducing substance use.
            (3) Link identified individuals to the nearest Level 2 facility.
            (4) Follow up monthly to monitor progress.
            (5) Mobilise community support groups.
            (6) Report aggregate data to the sub-county health team monthly.
            CHWs should NOT attempt detoxification or prescribe medication.
            CAGE questionnaire: Cut down? Annoyed? Guilty? Eye-opener?
            2+ yes answers = refer to health facility.
            """,
            "tags": ["CHW", "community", "CAGE", "household", "referral"]
        },
    ]

    return documents


def build_vector_store(documents: list):
    """
    Builds a ChromaDB vector store from the knowledge base documents.
    Uses multilingual sentence-transformers for Swahili-compatible embeddings.
    """
    print("Building vector store...")

    print("  Loading multilingual embedding model...")
    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("zeramatumizi_kb")
    except Exception:
        pass

    collection = client.create_collection(
        name="zeramatumizi_kb",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [doc["content"] for doc in documents]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=[doc["id"] for doc in documents],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"source": doc["source"], "tags": ", ".join(doc["tags"])}
            for doc in documents
        ]
    )

    print(f"Vector store built: {len(documents)} documents indexed")
    return collection, embedder


def retrieve_relevant_docs(query, collection, embedder, n_results=3):
    """
    Retrieves the most relevant documents for a query.
    Uses cosine similarity on multilingual embeddings.
    """
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "content": doc,
            "source": meta["source"],
            "relevance": round(1 - dist, 3)
        })

    return retrieved


def ask_gemini(query: str, context: str) -> str:
    """
    Sends the query and retrieved context to Groq's Llama model
    for a grounded response. Function name kept as ask_gemini
    for compatibility with the rest of the pipeline.
    """
    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "You are a ZeraMatumizi AI assistant supporting NACADA officers, "
        "school counsellors, and community health workers in Kenya.\n\n"
        "You answer questions about substance use disorders, intervention "
        "protocols, referral pathways, and community support - grounded "
        "strictly in the provided context from official Kenyan and WHO "
        "guidelines.\n\n"
        "Rules:\n"
        "- Answer in the same language as the question (Swahili or English)\n"
        "- Be practical and stepwise - give numbered steps where possible\n"
        "- Always mention the NACADA helpline 1192 when relevant\n"
        "- Never shame or stigmatise people with substance use disorders\n"
        "- If the answer is not in the context, say so honestly\n"
        "- Keep responses concise and actionable\n\n"
        f"Context from official protocols:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a clear, protocol-grounded response:"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": system_prompt}
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return response.choices[0].message.content


def run_rag_query(query, collection, embedder):
    """
    Full RAG pipeline: retrieve, augment, generate.
    """
    retrieved_docs = retrieve_relevant_docs(query, collection, embedder)

    context = "\n\n".join([
        f"[Source: {doc['source']} | Relevance: {doc['relevance']}]\n{doc['content']}"
        for doc in retrieved_docs
    ])

    response = ask_gemini(query, context)

    return {
        "query": query,
        "response": response,
        "sources": [doc["source"] for doc in retrieved_docs],
        "relevance_scores": [doc["relevance"] for doc in retrieved_docs],
    }


def print_rag_response(result):
    """Prints a formatted RAG response."""
    print("\n" + "=" * 60)
    print(f"QUERY: {result['query']}")
    print("=" * 60)
    print(f"\nRESPONSE:\n{result['response']}")
    print("\nSOURCES USED:")
    for source, score in zip(result["sources"], result["relevance_scores"]):
        print(f"  - {source} (relevance: {score})")
    print("=" * 60 + "\n")


def save_session_log(results):
    """Saves a session log of all queries and responses."""
    ensure_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(REPORTS_PATH, f"rag_session_{timestamp}.txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ZeraMatumizi RAG Pipeline Session Log\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        for result in results:
            f.write(f"QUERY: {result['query']}\n")
            f.write(f"RESPONSE:\n{result['response']}\n")
            f.write(f"SOURCES: {', '.join(result['sources'])}\n")
            f.write("-" * 60 + "\n\n")

    print(f"Session log saved: {filepath}")


if __name__ == "__main__":
    print("Initialising ZeraMatumizi RAG Pipeline...")
    ensure_directories()

    documents = create_knowledge_base()
    print(f"Knowledge base: {len(documents)} protocol documents loaded")

    collection, embedder = build_vector_store(documents)

    queries = [
        "A student appears to be using cannabis. What steps should I take?",
        "What is the referral pathway for severe alcohol use disorder in Kenya?",
        "Mwanafunzi anaonekana kutumia bhang. Ninafanya nini?",
        "Jinsi ya kuzungumza na mtu anayetumia chang'aa?",
        "How do I use the ASSIST screening tool?",
    ]

    results = []
    for query in queries:
        result = run_rag_query(query, collection, embedder)
        print_rag_response(result)
        results.append(result)

    save_session_log(results)

    print("RAG pipeline complete!")