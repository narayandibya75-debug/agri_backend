def recommend_fertilizer(N, P, K, ph=None):

    result = {}

    result["nitrogen"] = "Low → Add Urea" if N < 50 else "Excess → Avoid N" if N > 100 else "Optimal"
    result["phosphorus"] = "Low → Add DAP" if P < 50 else "Excess → Avoid P" if P > 100 else "Optimal"
    result["potassium"] = "Low → Add Potash" if K < 50 else "Excess → Avoid K" if K > 100 else "Optimal"

    if ph:
        result["ph"] = "Acidic → Add Lime" if ph < 5.5 else "Alkaline → Add Gypsum" if ph > 7.5 else "Optimal"

    result["overall"] = "Use balanced fertilizer (10-10-10)"

    return result