from PromptDAG import PromptDAG
from PromptACO import PromptACO
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Dataset import Dataset
from PromptEvaluator import PromptEvaluator
from StrategyACO import StrategyACO
import time


DEVICE = "mps"
MODEL_NAME = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
HUMAN_PROMPT = False


"""
PART OF A PROMPT:

RoleInstruction,
TaskInstruction,
FidelityConstraint,
SalienceGuidance,
StyleConstraint,
LengthConstraint,
Input,
OutputFormat
"""


def build_dataset(
    csv_path,
    input_col,
    output_col,
    n_samples=20,
    target_tokens=350,
    tolerance=20,
    train_fraction=0.8,
    seed=40
):
    dataset = Dataset(
        csv_path=csv_path,
        input_col=input_col,
        output_col=output_col,
    )

    dataset.load_with_token_limit(
        number_of_samples=n_samples,
        target_tokens=target_tokens,
        tolerance=tolerance
    )

    dataset.split_train_test(
        train_fraction=train_fraction,
        shuffle=True,
        seed=seed
    )

    return dataset


def build_prompt_dag(
    levels,
    base_texts,
    model,
    tokenizer,
    nodes_per_level,
    variants_csv="prompt_variants.csv"
):
    assert len(levels) == len(base_texts)
    assert len(levels) == len(nodes_per_level)

    dag = PromptDAG(levels)
    dag.initialize_levels(base_texts)

    dag.prepare_variants(
        load_variants=True,
        path_csv=variants_csv,
        nodes_per_level=list(nodes_per_level),
        model=model,
        tokenizer=tokenizer
    )

    dag.build_graph()
    return dag


def run_experiment(
    levels,
    base_texts,
    nodes_per_level,
    model,
    tokenizer,
    dataset_csv,
    results_csv,
    seed,
    strategyACO,
    strategy_name,
    n_ants=4,
    iterations=4,
    alpha=2,
    evaporation_rate=0.2
):
    random.seed(seed)

    dataset = build_dataset(
        csv_path=dataset_csv,
        input_col="article",
        output_col="highlights"
    )

    dag = build_prompt_dag(
        levels=levels,
        base_texts=base_texts,
        nodes_per_level=nodes_per_level,
        model=model,
        tokenizer=tokenizer
    )

    evaluator = PromptEvaluator(
        model,
        tokenizer,
        dataset.train_pairs
    )

    aco = PromptACO(
        dag=dag,
        evaluator=evaluator,
        strategyACO=strategyACO,
        strategy_name=strategy_name,
        n_ants=n_ants,
        iterations=iterations,
        alpha=alpha,
        evaporation_rate=evaporation_rate
    )

    best_prompt, best_score = aco.run()
    aco.save_history_to_csv(file_path=results_csv)

    return best_prompt, best_score, aco.history


EXPERIMENTS = [
    {
        "levels": ["TaskInstruction", "Input"],
        "base_texts": [
            "Summarize this text.",
            "the text:"
        ],
        "nodes_per_level": (15, 15),
        "strategy": StrategyACO(),
        "strategy_name": "EAS"
    },
    {
        "levels": ["TaskInstruction", "Input"],
        "base_texts": [
            "Summarize this text.",
            "the text:"
        ],
        "nodes_per_level": (15, 15),
        "strategy": StrategyACO(),
        "strategy_name": "Rank"
    },
    {
        "levels": ["TaskInstruction", "Input"],
        "base_texts": [
            "Summarize this text.",
            "the text:"
        ],
        "nodes_per_level": (15, 15),
        "strategy": StrategyACO(),
        "strategy_name": "MMAS"
    },
    {
        "levels": ["TaskInstruction", "Input"],
        "base_texts": [
            "Summarize this text.",
            "the text:"
        ],
        "nodes_per_level": (15, 15),
        "strategy": StrategyACO(),
        "strategy_name": "BWAS"
    }
]


def main():
    begin = time.time()
    base_seed = 40

    if not HUMAN_PROMPT:
        for idx, exp in enumerate(EXPERIMENTS):
            print(f"\nRunning experiment {idx + 1}/{len(EXPERIMENTS)}")
            print("Levels:", exp["levels"])
            print("Nodes per level:", exp["nodes_per_level"])
            print("Strategy:", exp["strategy_name"])

            run_experiment(
                levels=exp["levels"],
                base_texts=exp["base_texts"],
                nodes_per_level=exp["nodes_per_level"],
                model=model,
                tokenizer=tokenizer,
                dataset_csv="../data/test.csv",
                results_csv="../results/statsACO.csv",
                seed=base_seed,
                strategyACO=exp["strategy"],
                strategy_name=exp["strategy_name"]
            )

        print(f"\nTOTAL TIME TAKEN: {time.time() - begin} secs")
    else:

        prompt = "Summarize this text\nthe text : {INPUT}"
        dataset = build_dataset(csv_path="../data/test.csv",input_col="article",output_col="highlights")
        promptEvaluator = PromptEvaluator(model=model,tokenizer=tokenizer,dataset=dataset.train_pairs)
        promptEvaluator.evaluatePrompt(prompt=prompt,humanPrompt=HUMAN_PROMPT,humanPromptcsvPath="../results/statsHumanPrompt.csv")



if __name__ == "__main__":
    main()
