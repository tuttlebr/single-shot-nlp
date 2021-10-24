# PyTorch implimentation of the BERT model fine-tined on an abstract label task

Works by posing each candidate label as a "hypothesis" and the sequence which we want to classify as the "premise". This will softmax the scores for entailment vs. contradiction for each candidate label independently.

#### Zero-Shot multilingual inference engine

_Sample input which takes string and potential labels_

```bash
curl -X POST "http://HOSTIP:PORT/predict" \
            -H  "accept: application/json" \
            -H  "Content-Type: application/json" \
            --data-raw '{
                "input_string":"When thinking about automating developer workflows, the first things that come to mind for most are traditional CI/CD tasks: build, test, and deploy. However, many other common tasks can benefit from automation outside of traditional deployment pipelines.",
                "label_options":["laryngitis", "audit", "compliance", "security", "unicorns", "automation"]
                }'
```
