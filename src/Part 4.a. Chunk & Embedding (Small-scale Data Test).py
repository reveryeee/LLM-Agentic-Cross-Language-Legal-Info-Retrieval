def find_model_path():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'config.json' in files and 'jina' in root.lower():
            return root
    return None

LOCAL_MODEL_PATH = find_model_path()

# we use the RecursiveCharacterTextSplitter method for our chunking strategy
def perform_smart_chunking(df):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=400, separators=["\n\n", "\n", ". ", " ", ""]
    )
    final_texts = []
    final_metadatas = []
    
    for _, row in df.iterrows():
        cite = str(row['citation'])
        text = str(row['text'])
        
        # we keep a whole chunk if the token size is small, we split multiple chunks with RecursiveCharacterTextSplitter if token size is big 
        word_count = len(text.split())
        if word_count < 6000: 
            final_texts.append(text)
            final_metadatas.append(cite)
        else:
            chunks = splitter.split_text(text)
            for i, c in enumerate(chunks):
                final_texts.append(c)
                final_metadatas.append(f"{cite}#chunk{i}")
                
    return final_texts, final_metadatas


# We firstly test our retrieval (involve the text content and the citations graph relationship) workflow with a small amount of data here before we spend hours for our full dataset

def quick_verify_hybrid_search(sample_size=1000):
    
    df_map = pd.read_parquet(MAPPING_PATH)
    
    df_graph = pd.read_parquet(GRAPH_PATH)
  
    graph_dict = pd.Series(df_graph.refs.values, index=df_graph.citation).to_dict()
    
    df_sample = df_map.sample(min(sample_size, len(df_map)), random_state=42)
    test_texts, test_metadatas = perform_smart_chunking(df_sample)

    model = SentenceTransformer(LOCAL_MODEL_PATH, trust_remote_code=True, 
                                model_kwargs={"torch_dtype": torch.float16})
    if torch.cuda.is_available(): model.to('cuda')
    
    test_embeddings = model.encode(test_texts, batch_size=4, show_progress_bar=True, 
                                   normalize_embeddings=True, task="retrieval.passage")
    
    dim = test_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(test_embeddings.astype('float32'))
    
    test_query = "What happens to debt during a divorce?"
    print(f"\n User Query: '{test_query}'")
    
    query_emb = model.encode([test_query], task="retrieval.query", normalize_embeddings=True)
    distances, indices = index.search(query_emb.astype('float32'), k=3)
    
    print("\n" + "="*50)
    print("retrieval report")
    print("="*50)
    
    for i, idx in enumerate(indices[0]):
        raw_cite = test_metadatas[idx]
        score = distances[0][i]
        
        clean_id = normalize_citation(raw_cite)
        
        print(f"   [Rank {i+1}] similarity: {score:.4f}")
        print(f"   original ID: {raw_cite}")
        print(f"   normalized ID: {clean_id}")  
        
        linked_refs = graph_dict.get(clean_id, "")
        
        if linked_refs:
            refs_list = [r for r in linked_refs.split(';') if r]
            print(f"    graph successfully get {len(refs_list)} connected nodes:")
            print(f"       -> {', '.join(refs_list[:5])} ...") # we only show the top 5
        else:
            print(f"   No connected nodes for this node")
        print("-" * 30)


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    
    # check the model path for our environment (we use kaggle environment here)
    if not LOCAL_MODEL_PATH:
        print("No Jina model, please add the model first")
    else:
        build_infrastructure()
        
        quick_verify_hybrid_search(1000)
