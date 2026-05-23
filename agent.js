import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
  createToolCallingAgent,
  AgentExecutor,
} from "@langchain/classic/agents";
import { TavilySearch } from "@langchain/tavily";
import { createInterface } from "readline";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { createRetrievalChain } from "@langchain/classic/chains/retrieval";
import { TaskType } from "@google/generative-ai";

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { createRetrieverTool } from "@langchain/classic/agents/toolkits";

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

const vectorStores = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings,
);

//RETRIEVE DATA
const retriever = vectorStores.asRetriever({
  k: 2,
});

const model = new ChatGoogleGenerativeAI({
  model: "gemini-pro-latest",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant called Max."],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

// create and Assign tools
const searchTool = new TavilySearch({
  maxResults: 2,
  topic: "general",
});
const retrieverTool = createRetrieverTool(retriever, {
  name: "langchain_search",
  description: "Use this tool when searching for information about LangChain.",
});
const tools = [searchTool, retrieverTool];

// Create agent
const agent = await createToolCallingAgent({
  llm: model,
  prompt,
  tools,
});

// Create agent executor
const agentExecutor = new AgentExecutor({
  agent,
  tools,
});

// get user inpuiy
const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chatHistory = [];

const askQuestion = async () => {
  rl.question("User: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    // call agent
    const response = await agentExecutor.invoke({
      input,
      chat_history: chatHistory,
    });

    console.log("Agent: ", response.output);
    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(response.output));

    askQuestion();
  });
};

askQuestion();
