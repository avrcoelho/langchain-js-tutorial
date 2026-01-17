import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";

dotenv.config({ path: ".env" });

console.log(process.env.OPENAI_API_KEY);

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo",
  temperature: 0.7,
  maxTokens: 1_000,
  verbose: true,
});

model.invoke("Hello World").then((response) => {
  console.log(response);
});
