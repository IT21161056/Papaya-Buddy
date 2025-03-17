import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:mobile_app/models/treatmentModel.dart';
import 'package:mobile_app/utils/constants.dart';

class TreatmentService {
  static Future<List<Treatment>> getTreatmentDataByDisease(
      String diseaseId) async {
    final String url =
        '${BaseURL.BASE_URL}:5080/api/v1/treatment/by-disease/$diseaseId';

    try {
      print("Fetching from URL: $url");
      final response = await http.get(
        Uri.parse(url),
        headers: {'Content-Type': 'application/json'},
      );

      print("Response status code: ${response.statusCode}");
      print("Response body: ${response.body}");

      if (response.statusCode == 200) {
        // Check if the response body is empty
        if (response.body.isEmpty) {
          print("Response body is empty");
          return [];
        }

        final dynamic decodedData = json.decode(response.body);
        print("Decoded data type: ${decodedData.runtimeType}");

        if (decodedData == null) {
          print("Decoded data is null");
          return [];
        }

        if (decodedData is Map) {
          print("Available keys in response: ${decodedData.keys.toList()}");

          if (decodedData.containsKey('data')) {
            final data = decodedData['data'];
            print("Data type: ${data?.runtimeType}");

            if (data == null) {
              print("'data' field is null");
              return [];
            }

            if (data is List) {
              print("Data list length: ${data.length}");
              try {
                return data.map((item) => Treatment.fromJson(item)).toList();
              } catch (e) {
                print("Error mapping data to Treatment objects: $e");
                return [];
              }
            } else {
              print("'data' is not a List: $data");
              return [];
            }
          } else {
            print("Response does not contain 'data' key");
            return [];
          }
        } else {
          print("Decoded data is not a Map: $decodedData");
          return [];
        }
      } else {
        print("Failed to load treatments. Status code: ${response.statusCode}");
        print("Response body: ${response.body}");
        return [];
      }
    } catch (error) {
      print("Error fetching treatments: $error");
      print("Stack trace: ${StackTrace.current}");
      return [];
    }
  }
}
