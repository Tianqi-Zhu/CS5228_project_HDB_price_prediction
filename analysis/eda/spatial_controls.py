"""Run the complementary spatial studies through one canonical entry point."""

from __future__ import annotations


def main():
    from analysis.eda.analyze_amenities_adjusted import main as amenities
    from analysis.eda.analyze_mrt_adjusted import main as mrt_adjusted
    from analysis.eda.analyze_mrt_matched import main as mrt_matched
    from analysis.eda.analyze_spatial_features import main as overview
    from analysis.eda.spatial_common import load_spatial_training

    frame = load_spatial_training()
    overview(frame)
    mrt_adjusted(frame)
    mrt_matched(frame)
    amenities(frame)


if __name__ == "__main__":
    main()
