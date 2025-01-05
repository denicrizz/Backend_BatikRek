from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ml_model import search_by_suitable_for
import json

@csrf_exempt
def rekomendasi_batik(request):
    if request.method == "POST":
        try:
            # Parse data input dari request body
            data = json.loads(request.body)
            suitable_for_category = data.get("suitable_for", "").strip()  # Ambil kategori suitable_for
           
            if not suitable_for_category:
                return JsonResponse({"error": "suitable_for category is required"}, status=400)
            
            # Cari batik berdasarkan kategori suitable_for
            results = search_by_suitable_for(suitable_for_category).to_dict(orient="records")
            
            if results:
                return JsonResponse({"recommended_batik": results}, status=200)
            else:
                return JsonResponse({"message": "No matching batik found for the given category"}, status=404)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method. Only POST is allowed."}, status=400)
