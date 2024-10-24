# 🔍 Fakt.ai

*Fully transparent, automated fact-checking powered by AI Agents*

Try it for free here: https://faktai.streamlit.app/

## ⚠️ The Problem

* Have you ever read something online and thought "I don’t know if that’s true"?
* Have you ever heard someone confidently speaking about a tense topic? They quote research 
  papers saying you can "look it up" and "read them yourself".
* Have you ever actually tried to look them up?

I did. And hit two major obstacles:

1. Research is hard to find.
2. Research is hard to understand. Does a paper saying they observed a 6% uptake in  
   protein R-X-517A across all participants support what the original speaker was saying?

## ✅ The Solution

We need a tool that:

* Finds research papers that support/contradict given arguments
* Analyses said papers for quality (e.g. are they biased lobbying groups, is the sample size too 
  small?)
* Compares and contrasts arguments to conclude whether the original claim is true
* Is fully transparent, so the user can see each step of the way

Enter Fakt AI - fully transparent, automated fact-checking powered by AI Agents. 


## ⚙️ Installation

1. Install [Poetry](https://python-poetry.org/)
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. cd into the project folder and install dependencies with
    ```bash
    poetry install
    ```

3. Create an `.env` file and add the follwing API keys
    ```
    GROQ_API_KEY=...              (required)
    SEMANTIC_SCHOLAR_API_KEY=...  (optional, results in more successful runs)
    ```
    - Grok: Sign up to [Grok Cloud](https://console.groq.com/login) and create an API key
    - Semantic Scholar: follow the instructions [here](https://www.semanticscholar.org/product/api#api-key) (it will take a few days to arrive, annoyingly)
4. Run a query (takes 1-2 minutes)
   ```bash
   poetry run python main.py --query "were treatments available to treat COVID before the vaccines came out?"
   ```
Note: at the moment, Fakt AI only supports academic paper search. So please ask questions that require academic papers to answer them.

Note 2: it can be a bit tempermental. If you do not get back an answer, please re-run the query. Usually it works the second time.

## 💪 Areas for Improvement / Roadmap

* Deploy as a front-end for nicer user interactions
* Clickable URL links for every reference
* Support for more than just academic paper search
* Run analysis on every reference (e.g. bias likelihood check, analyse any experiments in the paper, is the sample size too small etc.)
* Chat with the agent after e.g. ask questions about specific bits of the process or to search more in certain areas
* Query optimisation/breakdown. Breakdown queries into multiple steps and perform multiple searches to improve results
* Faster!
* A nice UI to be able to step back through the thinking and easily open up papers and see relevant sections
* Handle questions that even the experts have not come to a conclusion about e.g. 60% of the data supports this PoV, 40% supports the opposite.
  
## 💰 Monetisation

* Pay per fact check - powered by the FaktAI token

Who?

* Journalists
* Academics
* Researchers
* General people who want to be better informed / know the truth e.g. watch Russel Brand, use X, follow Robert F Kennedy.
