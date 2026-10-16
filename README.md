# SmartRent

Rent prediction for Indian cities (Mumbai, Pune, Bangalore, New Delhi, Nagpur).

```
python -m ml.training.train      # trains model -> ml/models/
uvicorn api.main:app --reload    # open http://127.0.0.1:8000
```
