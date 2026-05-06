import { ChatGoogle } from "@langchain/google";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from "dotenv";

dotenv.config({ path: ".env", override: true });

// Create model
const model = new ChatGoogle({
  model: "gemini-pro-latest",
  temperature: 0.7,
});

// Create prompt template
// const prompt = ChatPromptTemplate.fromTemplate(
//   "You are a comedian. Tell me a joke about {topic}.",
// );
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "Generate a joke about a given topic."],
  ["user", "Tell me a joke about {topic}."],
]);
// console.log(await prompt.format({ topic: "chicken" }));

// Create chain
const chain = prompt.pipe(model);

// call response
const response = await chain.invoke({ topic: "chicken" });
console.log(response.content);
