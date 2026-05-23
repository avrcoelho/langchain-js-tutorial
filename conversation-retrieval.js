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
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "@langchain/classic/chains/history_aware_retriever";
// Initialize environment variables first
dotenv.config({ path: ".env", override: true });

// Load data and create vector store
const createVectorStore = async () => {
  const loader = new CheerioWebBaseLoader("https://docs.langchain.com/");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "gemini-embedding-001", // Google's standard text embedding model
    taskType: TaskType.RETRIEVAL_DOCUMENT,
  });

  return MemoryVectorStore.fromDocuments(splitDocs, embeddings);
};

// Custom implementation of createStuffDocumentsChain using LCEL

// Create retrieval chain
const createChain = async (vectorStores) => {
  const model = new ChatGoogle({
    model: "gemini-pro-latest",
    temperature: 0.7,
  });

  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "Answer the user`s question based on the following context: {context}.",
    ],
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
  ]);

  // const chain = prompt.pipe(model);
  const combineDocsChain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  //RETRIEVE DATA
  const retriever = vectorStores.asRetriever({
    k: 2,
  });

  const retrieverPropmpt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder("chat_history"),
    ["user", "{input}"],
    [
      "user",
      "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.",
    ],
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPropmpt,
  });

  // Custom retrieval chain to bypass the `@langchain/classic` `isBaseRetriever` bug
  return RunnableSequence.from([
    RunnablePassthrough.assign({
      context: historyAwareRetriever.withConfig({
        runName: "retrieve_documents",
      }),
      chat_history: (input) => input.chat_history ?? [],
    }),
    RunnablePassthrough.assign({
      answer: combineDocsChain,
    }),
  ]).withConfig({ runName: "retrieval_chain" });
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

// chat history
const chatHistory = [
  new HumanMessage("Hello"),
  new AIMessage("Hi, how can I help you?"),
  new HumanMessage("My name is André"),
  new AIMessage("Hi André, how can I help you?"),
  new HumanMessage("What is LangChain?"),
  new AIMessage(
    "Langchain is an open-source framework designed to simplify the development of applications powered by Large Language Models (LLMs) like OpenAI's GPT-4, Llama 2, and others. ",
  ),
];

const response = await chain.invoke({
  input: "What is my name?",
  chat_history: chatHistory,
});

console.log(response);
