1) Insert species/kw questions into the database `python insert_species_kw_questions_into_db.py path_to_question_file.txt`
2) Search for links for those questions/kewwords with `python dataforseo_search_pipeline.py`
3) Extract the content from those links using the instructions in the docker-compose file paird with the `scrape_links_content.py` script.
4) Do a quick pass on the content for validity with a cheap and fast model using `classify_content_as_valid.py`
5) Answer questions using the `answer_questions_with_content.py` script.

Tools:
- `report_missing_species_questions.py` -- lists all the questions sorted by species that do not have hyperlinks discovered by a search engine links in the database.
