from main import predict_sms, SMSRequest, load_model

def run_tests():
    print("\n" + "="*50)
    print("🛡️  CHECKLEY AI - SCAM DETECTION TEST SUITE  🛡️")
    print("="*50 + "\n")
    
    # Simulate FastAPI startup event
    load_model()
    print("-" * 50)

    test_cases = [
        {
            "type": "Palestinian Safe",
            "text": "يعطيك العافية يابا، لا تنسى تجيب الخبز معك وانت مروح"
        },
        {
            "type": "Palestinian Scam",
            "text": "تم تجميد حسابك البنكي في بنك فلسطين. ادخل الرابط حالا لتجنب الغرامة او الحظر bop.link/12"
        },
        {
            "type": "General Scam (Prize)",
            "text": "مبروك ربحت سيارة جيب من جوال، اضغط هنا لتسجيل الاستلام"
        },
        {
            "type": "Formal Safe",
            "text": "نذكركم بموعد اجتماع الغد في تمام الساعة العاشرة صباحا"
        },
        {
            "type": "Social Engineering",
            "text": "انا بالغلط بعتلك كود ممكن تعطيني اياه؟ ضروري"
        }
    ]

    for ct, case in enumerate(test_cases, 1):
        # Create the exact payload the API expects
        payload = SMSRequest(text=case["text"])
        
        # Hit the prediction logic
        try:
            res = predict_sms(payload)
            print(f"[{ct}] Category: {case['type']}")
            print(f"    Message:  {case['text']}")
            
            # Print visually distinct results depending on scam vs safe
            if res.is_scam:
                print(f"    Result:   ❌ SCAM / HIGH RISK")
            else:
                print(f"    Result:   ✅ SAFE")
                
            print(f"    Details:  {res.message} (Confidence: {res.confidence}%)")
            print("-" * 50)
        except Exception as e:
            print(f"[{ct}] Error evaluating message: {e}")

if __name__ == "__main__":
    run_tests()
