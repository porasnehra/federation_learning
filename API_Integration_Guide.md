# Mule Detection API Integration Guide (Flutter)

This document provides everything the Flutter developer needs to connect the frontend application to the Mule Account Detection backend model deployed on Render.

## 1. Base URL
Your backend is hosted on Render. Once the service is live, you will have a public URL.
**Base URL:** `https://<your-render-url>.onrender.com`
*(Replace `<your-render-url>` with the actual slug provided by Render).*

---

## 2. Endpoints

### Health Check
Check if the API is awake and running.
*   **Method:** `GET`
*   **Path:** `/`
*   **Response:**
    ```json
    {
      "status": "Active",
      "message": "Mule Detection API is running. Send POST request to /predict"
    }
    ```

### Evaluate Account Risk (Prediction)
Send account transaction details to get a mule risk prediction.
*   **Method:** `POST`
*   **Path:** `/predict`
*   **Headers:**
    *   `Content-Type: application/json`

#### Request Payload
The backend expects a JSON object containing a `features` dictionary. The keys in this dictionary must map to the raw dataset columns expected by the ML pipeline.

**Example Request Body:**
```json
{
  "features": {
    "F3799": 150000.0,    // Total inward amount (INR)
    "F3800": 14000.0,     // Total outward amount (INR)
    "F3796": 45,          // Number of credit transactions
    "F3797": 12,          // Number of debit transactions
    "F3891": "student",   // Occupation (e.g., student, salaried, housewife, others)
    "F3886": "Savings",   // Account Type (e.g., Savings, Current, MSME Medium)
    "F3889": "L7D",       // Tenure/SIM Age (e.g., L7D, L14D, L30D, G365D)
    "F3890": "U",         // Geographic Zone (R, SU, U, M)
    "F3893": "RETAIL",    // Segment (RETAIL, CORPORATE)
    "F3894": 22,          // Age
    "F3895": 450,         // Credit Score
    "F3887": 12,          // Tenure in months
    "F3919": 1            // Linked accounts count
  }
}
```
*(Note: Any missing features not sent by the frontend will automatically default to `0` or baseline values in the backend, so it is flexible! Send as much data as you have).*

#### Response Payload
The API returns a JSON object containing the probability score, risk tier, and any specific fraud signals triggered.

**Example Response:**
```json
{
  "mule_probability": 0.9103,
  "risk_tier": "CRITICAL",
  "flagged": true,
  "signals_triggered": 2,
  "signals": [
    "Signal 6: Student Account High Velocity",
    "Signal 11: New Account (SIM/Telecom Age)"
  ]
}
```

**Risk Tiers Available:**
*   `CRITICAL` (Probability ≥ 0.80)
*   `HIGH` (Probability ≥ 0.50)
*   `MEDIUM` (Probability ≥ 0.35)
*   `LOW` (Probability < 0.35)

---

## 3. Flutter Implementation Example

Here is a ready-to-use Dart function for the Flutter application using the `http` package:

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class MuleDetectionService {
  // Replace with your actual Render URL
  static const String baseUrl = 'https://YOUR_APP_NAME.onrender.com';

  /// Evaluates an account for mule risk
  static Future<Map<String, dynamic>> checkAccountRisk(Map<String, dynamic> collectedData) async {
    final url = Uri.parse('$baseUrl/predict');
    
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        // Wrap the collected data inside the "features" key
        body: jsonEncode({"features": collectedData}),
      );

      if (response.statusCode == 200) {
        // Success
        return jsonDecode(response.body);
      } else {
        // Handle API Error
        throw Exception('Failed to load risk analysis: ${response.statusCode}');
      }
    } catch (e) {
      // Handle Network/Timeout Error
      throw Exception('Network error occurred: $e');
    }
  }
}
```

### Usage in UI
```dart
void onSubmit() async {
  Map<String, dynamic> userData = {
    "F3799": 150000,
    "F3891": "student",
    "F3894": 22,
  };

  try {
    final result = await MuleDetectionService.checkAccountRisk(userData);
    
    print("Risk Tier: ${result['risk_tier']}");           // e.g., CRITICAL
    print("Probability: ${result['mule_probability']}");  // e.g., 0.9103
    print("Flagged: ${result['flagged']}");               // e.g., true
    print("Signals: ${result['signals']}");               // e.g., ['Signal 6...', 'Signal 11...']
    
    // TODO: Update UI based on the risk_tier
  } catch (error) {
    print(error);
  }
}
```
