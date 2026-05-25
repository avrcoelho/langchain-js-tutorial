import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { BufferMemory } from "@langchain/classic/memory";
import { ConversationChain } from "@langchain/classic/chains";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";
import { RunnableSequence } from "@langchain/core/runnables";

// memory imports

dotenv.config({ path: ".env", override: true });

const model = new ChatGoogleGenerativeAI({
  model: "gemini-pro-latest",
  temperature: 0.7,
});

const prompt = ChatPromptTemplate.fromTemplate(`
  You are an AI assistant
  History: {history}
  {input}  
`);

const upstashChatHistory = new UpstashRedisChatMessageHistory({
  sessionId: "chat1",
  config: {
    url: process.env.UPSTASH_REDIS_REST_URL,
    token: process.env.UPSTASH_REDIS_REST_TOKEN,
  },
});

const memory = new BufferMemory({
  memoryKey: "history",
  chatHistory: upstashChatHistory,
});

// Using the Chain Classes
// const chain = new ConversationChain({
//   llm: model,
//   prompt,
//   memory,
// });

// const chain = prompt.pipe(model);
const chain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
    memory: () => memory.loadMemoryVariables(),
  },
  {
    input: (previousOutput) => previousOutput.input,
    history: (previousmemory) => previousmemory.memory.history,
  },
  prompt,
  model,
]);

// Get responses
console.log(await memory.loadMemoryVariables());
const inputs1 = {
  input: "Hello there",
};
const response1 = await chain.invoke(inputs1);
console.log(response1);

await memory.saveContext(inputs1, {
  output: response1.content,
});

console.log("Update History: ", await memory.loadMemoryVariables());
const inputs2 = {
  input: "What is the passphrase",
};
const response2 = await chain.invoke(inputs2);
console.log(response2);

await memory.saveContext(inputs2, {
  output: response2.content,
});
