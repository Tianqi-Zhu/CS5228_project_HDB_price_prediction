from pathlib import Path

from evaluator import HDBEvaluator
from preprocessor import HDBPreprocessor, PreprocessorConfig
from trainer import HDBTrainer
from visualizer import HDBVisualizer


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
AUXILIARY_DIR = DATA_DIR / "auxiliary"

TARGET_COL = "monthly_rent"


def main():
    # 1. Load training data
    preprocessor = HDBPreprocessor(
        PreprocessorConfig(
            target_col=TARGET_COL,
            # date_col="rent_date",
        )
    )

    train_df = preprocessor.load_data(TRAIN_PATH)

    # Optional competition / held-out test set.
    test_df = None
    if TEST_PATH.exists():
        test_df = preprocessor.load_data(TEST_PATH)

    # Auxiliary datasets can be loaded as needed, for example:
    #
    # mrt_df = preprocessor.load_data(AUXILIARY_DIR / "mrt.csv")
    # school_df = preprocessor.load_data(AUXILIARY_DIR / "schools.csv")
    #
    # Merge auxiliary features into train_df / test_df inside a dedicated
    # feature-engineering step once the auxiliary schemas are known.

    # 2. Basic EDA
    eda = preprocessor.basic_eda(train_df)
    print("Training shape:", eda["shape"])
    print("Duplicate rows:", eda["duplicates"])

    train_df = preprocessor.clean(train_df)
    train_df = preprocessor.engineer_features(train_df)

    if test_df is not None:
        test_df = preprocessor.clean(test_df)
        test_df = preprocessor.engineer_features(test_df)

    # 3. Visualize training data
    visualizer = HDBVisualizer()
    visualizer.generate_default_eda_plots(
        train_df,
        target_col=TARGET_COL,
    )

    # 4. Prepare model inputs
    X, y = preprocessor.split_features_target(train_df)
    transformer = preprocessor.build_transformer(X)

    # 5. Internal train/validation split + training
    trainer = HDBTrainer()
    X_train, X_valid, y_train, y_valid = trainer.split_data(X, y)

    pipeline = trainer.build_pipeline(transformer)
    model = trainer.train(pipeline, X_train, y_train)

    trainer.save_model(model)

    # 6. Evaluate on validation split
    evaluator = HDBEvaluator()
    metrics = evaluator.evaluate(model, X_valid, y_valid)
    evaluator.print_report(metrics)

    predictions = evaluator.prediction_frame(
        model,
        X_valid,
        y_valid,
    )
    predictions.to_csv(
        "reports/validation_predictions.csv",
        index=False,
    )

    # 7. Optional predictions for external test.csv
    if test_df is not None:
        test_predictions = model.predict(test_df)
        output_path = Path("reports/test_predictions.csv")

        test_output = test_df.copy()
        test_output["predicted_monthly_rent"] = test_predictions
        test_output.to_csv(output_path, index=False)

        print(f"Saved external test predictions to {output_path}")


if __name__ == "__main__":
    main()
