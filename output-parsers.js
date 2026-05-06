import { ChatGoogle } from "@langchain/google";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import dotenv from "dotenv";
import {
  CommaSeparatedListOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { z } from "zod";

dotenv.config({ path: ".env", override: true });

// Create model
const model = new ChatGoogle({
  model: "gemini-pro-latest",
  temperature: 0.7,
});

async function callStringOutputParser() {
  // Create prompt template
  // const prompt = ChatPromptTemplate.fromTemplate(
  //   "You are a comedian. Tell me a joke about {topic}.",
  // );
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Generate a joke about a given topic."],
    ["user", "Tell me a joke about {topic}."],
  ]);
  // console.log(await prompt.format({ topic: "chicken" }));

  // Create parser
  const parser = StringOutputParser();

  // Create chain
  const chain = prompt.pipe(model).pipe(parser);

  // call response
  return chain.invoke({ topic: "cat" });
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(
    'Provide 5 synonyms, separated by commas, for the word "{word}".',
  );
  const outoutputParser = CommaSeparatedListOutputParser();
  const chain = prompt.pipe(model).pipe(outoutputParser);
  return chain.invoke({ word: "happy" });
}

// const response = await callListOutputParser();

// console.log(response);

// Structured output parser
async function callStructuredOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract infromation from the following phrase.
    Formatting instructions: {formatting_instructions}
    Phrase: "{phrase}"
    `);
  const outputParser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "the name of the person",
    age: "age of the person",
  });
  const chain = prompt.pipe(model).pipe(outputParser);
  return chain.invoke({
    phrase: "My name is John and I am 30 years old.",
    formatting_instructions: outputParser.getFormatInstructions(),
  });
}

async function callZodOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Extract infromation from the following phrase.
    Formatting instructions: {formatting_instructions}
    Phrase: "{phrase}"
    `);
  const outputParser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("The name of the recipe"),
      ingredients: z
        .array(z.string())
        .describe("A list of ingredients needed for the recipe"),
    }),
  );
  const chain = prompt.pipe(model).pipe(outputParser);
  return chain.invoke({
    phrase:
      "I want to make spaghetti. I need pasta, tomato sauce, and meatballs.",
    formatting_instructions: outputParser.getFormatInstructions(),
  });
}
