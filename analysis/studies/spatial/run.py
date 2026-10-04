"""Run the complementary spatial studies through one canonical entry point."""

from __future__ import annotations


def main():
    from analysis.studies.spatial.amenities_fixed_effects import main as amenities
    from analysis.studies.spatial.mrt_fixed_effects import main as mrt_adjusted
    from analysis.studies.spatial.mrt_matching import main as mrt_matched
    from analysis.studies.spatial.overview import main as overview
    from analysis.studies.spatial.common import load_spatial_training

    frame = load_spatial_training()
    overview(frame)
    mrt_adjusted(frame)
    mrt_matched(frame)
    amenities(frame)


if __name__ == "__main__":
    main()
