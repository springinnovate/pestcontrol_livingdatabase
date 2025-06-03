"""Ask a quick question."""

import argparse
import asyncio
import trait_search_pipeline


async def main():
    """Entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, help="Answer context text")
    parser.add_argument("--query", type=str, help="Query text")

    args = parser.parse_args()
    openai_semaphore = asyncio.Semaphore()
    result = await trait_search_pipeline.get_webpage_answers(
        openai_semaphore,
        args.context,
        args.query,
        None,
    )

    print(f"answer: {result}")


if __name__ == "__main__":
    asyncio.run(main())
