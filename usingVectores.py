from pymilvus import MilvusClient

client = MilvusClient("./milvus_demo.db")


client.create_collection(collection_name="Faces",dimension=128)

