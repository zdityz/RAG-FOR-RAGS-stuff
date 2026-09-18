import sys
import json
from .pipeline import RAGPipeline


def main():
    print("Testing Full Multi-Agent RAG Pipeline...")
    test_query = input("\nEnter a question: ").strip()
    
    if not test_query:
        sys.exit(0)
        
    pipeline = RAGPipeline()
    result = pipeline.run(test_query)
    print("\n--- RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()