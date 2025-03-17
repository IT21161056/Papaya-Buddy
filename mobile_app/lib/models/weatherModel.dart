// class Weather {
//   final String cityName;
//   final double temperature;
//   final double feelsLike;
//   final double tempMin;
//   final double tempMax;
//   final int humidity;
//   final double windSpeed;
//   final int uvIndex;
//   final String description;
//   final int conditionCode;

//   Weather({
//     required this.cityName,
//     required this.temperature,
//     required this.feelsLike,
//     required this.tempMin,
//     required this.tempMax,
//     required this.humidity,
//     required this.windSpeed,
//     required this.uvIndex,
//     required this.description,
//     required this.conditionCode,
//   });

//   factory Weather.fromJson(Map<String, dynamic> json) {
//     return Weather(
//       cityName: json['name'] ?? '',
//       temperature: (json['main']['temp'] as num).toDouble(),
//       feelsLike: (json['main']['feels_like'] as num).toDouble(),
//       tempMin: (json['main']['temp_min'] as num).toDouble(),
//       tempMax: (json['main']['temp_max'] as num).toDouble(),
//       humidity: json['main']['humidity'] as int,
//       windSpeed: (json['wind']['speed'] as num).toDouble(),
//       uvIndex: json['uvi'] != null ? (json['uvi'] as num).toInt() : 0,
//       description: json['weather'][0]['description'],
//       conditionCode: json['weather'][0]['id'],
//     );
//   }

//   String getFormattedTemperature() {
//     return '${temperature.toStringAsFixed(1)}°C';
//   }

//   String getTemperatureRange() {
//     return '${tempMin.toStringAsFixed(0)}°C / ${tempMax.toStringAsFixed(0)}°C';
//   }

//   String getUvIndexCategory() {
//     if (uvIndex <= 2) return 'Low';
//     if (uvIndex <= 5) return 'Moderate';
//     if (uvIndex <= 7) return 'High';
//     if (uvIndex <= 10) return 'Very High';
//     return 'Extreme';
//   }
// }
