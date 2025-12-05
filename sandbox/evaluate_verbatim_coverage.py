import pandas as pd
from ariadne.verbatim_mapping.vocab_verbatim_term_mapper import VocabVerbatimTermMapper

mapper = VocabVerbatimTermMapper()

terms = pd.read_csv("E:/temp/mapping_quality/ICD10CMterms.csv")
mapped_count = 0
unmapped_count = 0
count = 0
concept_ids = []
concept_names = []
for term in terms["concept_name"].tolist():
    concepts = mapper.map_term(term)
    if concepts:
        mapped_count += 1
        concept_ids.append(";".join(str(c[0]) for c in concepts))
        concept_names.append(";".join(c[1] for c in concepts))
    else:
        unmapped_count += 1
        concept_ids.append("")
        concept_names.append("")
    count += 1
    if count % 1000 == 0:
        print(f"Processed {count} terms...")
terms["mapped_concept_ids"] = concept_ids
terms["mapped_concept_names"] = concept_names
terms.to_csv("E:/temp/mapping_quality/ICD10CMterms_mapped.csv", index=False)
print(f"Mapped terms: {mapped_count}, Unmapped terms: {unmapped_count}")
