import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatGoogle } from "@langchain/google";
import dotenv from "dotenv";
import { Document } from "@langchain/core/documents";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import * as cheerio from "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";

// Initialize environment variables first
dotenv.config({ path: ".env", override: true });

const embeddings = new GoogleGenerativeAIEmbeddings({
  modelName: "gemini-embedding-001", // Google's standard text embedding model
  taskType: TaskType.RETRIEVAL_DOCUMENT,
});

const loader = new CheerioWebBaseLoader("https://docs.langchain.com/");
const docs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);

// Custom implementation of createStuffDocumentsChain using LCEL
const createStuffDocumentsChain = async ({ llm, prompt }) => {
  return RunnableSequence.from([
    RunnablePassthrough.assign({
      context: (input) => {
        const docs = input.context;
        if (!Array.isArray(docs)) {
          return "";
        }
        return docs.map((doc) => doc.pageContent).join("\n\n");
      },
    }),
    prompt,
    llm,
  ]);
};



const model = new ChatGoogle({
  model: "gemini-pro-latest",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(
  `Answer the user question: {input}
  Context: {context}
  Question: {input}
  `,
);

// const chain = prompt.pipe(model);
const combineDocsChain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

// Documents
const documentA = new Document({
  pageContent: "Paris is the capital of France.",
});

const documentB = new Document({
  pageContent: "London is the capital of England.",
});

const vectorStores = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

//RETRIEVE DATA
const retriever = vectorStores.asRetriever({
  k: 2,
});

const retrievalChain = await createRetrievalChain({
  combineDocsChain,
  retriever,
});

const response = await retrievalChain.invoke({
  input: "What is LangChain?",
});

console.log(response.answer.content);
