// import 'dart:async';
// import 'package:flutter/material.dart';
// import 'package:flutter_svg/flutter_svg.dart';
// import 'package:intl/intl.dart';
// import 'package:mobile_app/models/weatherModel.dart';
// import 'package:mobile_app/theme/colors.dart';
// import 'package:mobile_app/widgets/weather.widgets/weather_info.dart';
// import '../../services/weatherService.dart';
// import '../../utils/constants.dart';

// class ExpandableWeatherCard extends StatefulWidget {
//   const ExpandableWeatherCard({super.key});

//   @override
//   State<ExpandableWeatherCard> createState() => _ExpandableWeatherCardState();
// }

// class _ExpandableWeatherCardState extends State<ExpandableWeatherCard> {
//   // API key for OpenWeatherMap
//   final _weatherService = WeatherService(ApiConstants.OPENWEATHERMAP_API_KEY);
//   Weather? _weather;
//   Timer? _weatherUpdateTimer;
//   bool isExpanded = false;

//   // fetch weather data
//   Future<void> _fetchWeatherData() async {
//     // Get the current city
//     final cityName = await _weatherService.getCurrentCity();

//     // Get weather data for the current city
//     try {
//       final weather = await _weatherService.getCurrentWeather(cityName);
//       setState(() {
//         _weather = weather;
//       });
//     } catch (e) {
//       print('Error: $e');
//     }
//   }

//   @override
//   void initState() {
//     super.initState();

//     // Fetch weather on startup
//     _fetchWeatherData();

//     // Set timer to fetch weather every 15 minutes
//     _weatherUpdateTimer = Timer.periodic(const Duration(minutes: 10), (_) {
//       _fetchWeatherData();
//     });
//   }

//   @override
//   void dispose() {
//     // Cancel the timer when the widget is disposed
//     _weatherUpdateTimer?.cancel();
//     super.dispose();
//   }

//   String getFormattedDate() {
//     DateTime now = DateTime.now();
//     String formattedDate = DateFormat('EEE, MMM d').format(now);
//     return 'Today, $formattedDate';
//   }

//   // Format the weather description to capitalize first letter of each word
//   String formatDescription(String description) {
//     if (description.isEmpty) return '';

//     return description.split(' ').map((word) {
//       if (word.isEmpty) return '';
//       return word[0].toUpperCase() + word.substring(1);
//     }).join(' ');
//   }

//   // Get the weather icon based on the condition code
//   String getWeatherIcon() {
//     if (_weather == null) {
//       return 'assets/icons/lucide_sun.svg';
//     }

//     final conditionCode = _weather!.conditionCode;
//     final iconName = getIconNameFromConditionCode(conditionCode);

//     return 'assets/icons/$iconName.svg';
//   }

//   // Get icon color based on weather condition
//   Color getWeatherIconColor() {
//     if (_weather == null) {
//       return Colors.amber;
//     }

//     final conditionCode = _weather!.conditionCode;

//     // Thunderstorm
//     if (conditionCode >= 200 && conditionCode < 300) {
//       return Colors.deepPurple;
//     }
//     // Drizzle or Rain
//     else if ((conditionCode >= 300 && conditionCode < 400) ||
//         (conditionCode >= 500 && conditionCode < 600)) {
//       return Colors.blue;
//     }
//     // Snow
//     else if (conditionCode >= 600 && conditionCode < 700) {
//       return Colors.lightBlue;
//     }
//     // Atmosphere (fog, mist, etc.)
//     else if (conditionCode >= 700 && conditionCode < 800) {
//       return Colors.blueGrey;
//     }
//     // Clear
//     else if (conditionCode == 800) {
//       return Colors.amber;
//     }
//     // Clouds
//     else {
//       return Colors.grey;
//     }
//   }

//   // Get icon background color
//   Color getWeatherIconBgColor() {
//     final iconColor = getWeatherIconColor();
//     return iconColor;
//   }

//   // Map OpenWeatherMap condition codes to icon names
//   String getIconNameFromConditionCode(int conditionCode) {
//     // Thunderstorm
//     if (conditionCode >= 200 && conditionCode < 300) {
//       return 'lucide_cloud-lightning';
//     }
//     // Drizzle
//     else if (conditionCode >= 300 && conditionCode < 400) {
//       return 'lucide_cloud-drizzle';
//     }
//     // Rain
//     else if (conditionCode >= 500 && conditionCode < 600) {
//       return 'lucide_cloud-rain';
//     }
//     // Snow
//     else if (conditionCode >= 600 && conditionCode < 700) {
//       return 'lucide_cloud-snow';
//     }
//     // Atmosphere (fog, mist, etc.)
//     else if (conditionCode >= 700 && conditionCode < 800) {
//       return 'lucide_cloud-fog';
//     }
//     // Clear
//     else if (conditionCode == 800) {
//       return 'lucide_sun';
//     }
//     // Clouds
//     else {
//       return 'lucide_cloud';
//     }
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Container(
//       padding: const EdgeInsets.all(20),
//       decoration: BoxDecoration(
//         color: Colors.white,
//         borderRadius: const BorderRadius.only(
//           bottomLeft: Radius.circular(24),
//           bottomRight: Radius.circular(24),
//         ),
//         boxShadow: [
//           BoxShadow(
//             color: Colors.black.withOpacity(0.05),
//             blurRadius: 5,
//             offset: const Offset(0, 2),
//           ),
//         ],
//       ),
//       child: GestureDetector(
//         onTap: () {
//           setState(() {
//             isExpanded = !isExpanded;
//           });
//         },
//         child: Container(
//           padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
//           decoration: BoxDecoration(
//             color: const Color(0xFFF8FAFC),
//             borderRadius: BorderRadius.circular(12),
//           ),
//           child: Column(
//             children: [
//               Row(
//                 children: [
//                   Container(
//                     decoration: BoxDecoration(
//                       shape: BoxShape.circle,
//                       color: Colors.white,
//                     ),
//                     child: Padding(
//                       padding: const EdgeInsets.all(10.0),
//                       child: SvgPicture.asset(
//                         getWeatherIcon(),
//                         height: 24,
//                         width: 24,
//                         color: getWeatherIconColor(),
//                       ),
//                     ),
//                   ),
//                   const SizedBox(width: 12),
//                   Column(
//                     crossAxisAlignment: CrossAxisAlignment.start,
//                     children: [
//                       if (_weather != null)
//                         Text(
//                           formatDescription(_weather?.description ?? ""),
//                           style: const TextStyle(
//                               fontSize: 12,
//                               color: Color.fromRGBO(27, 106, 216, 1),
//                               fontWeight: FontWeight.w500),
//                           maxLines: 1,
//                           overflow: TextOverflow.ellipsis,
//                         ),
//                       Text(
//                         getFormattedDate(),
//                         style: TextStyle(
//                           fontSize: 14,
//                           color: Color.fromRGBO(100, 116, 139, 1),
//                         ),
//                       ),
//                       Text(
//                         _weather != null
//                             ? '${_weather?.getFormattedTemperature()}'
//                             : 'Loading Temperature...',
//                         style: TextStyle(
//                           fontSize: 16,
//                           fontWeight: FontWeight.w600,
//                         ),
//                       ),
//                       const SizedBox(height: 4),
//                       Row(
//                         children: [
//                           const Icon(Icons.location_on,
//                               size: 16, color: Colors.red),
//                           const SizedBox(width: 4),
//                           Text(
//                             _weather?.cityName ??
//                                 "Loading City...", // Replace with dynamic location
//                             style: const TextStyle(
//                               fontSize: 14,
//                               color: Color.fromRGBO(100, 116, 139, 1),
//                             ),
//                           ),
//                         ],
//                       ),
//                     ],
//                   ),
//                   const Spacer(),
//                   Icon(
//                     isExpanded ? Icons.expand_less : Icons.expand_more,
//                     size: 28,
//                   ),
//                 ],
//               ),

//               // Expanded Content
//               AnimatedSize(
//                   duration: const Duration(milliseconds: 300),
//                   curve: Curves.easeInOut,
//                   child: isExpanded
//                       ? Column(
//                           children: [
//                             const SizedBox(height: 8),
//                             const Divider(
//                               color: Color.fromRGBO(226, 232, 240, 1),
//                             ),
//                             const SizedBox(height: 8),
//                             Row(
//                               mainAxisAlignment: MainAxisAlignment.spaceAround,
//                               children: [
//                                 WeatherInfo(
//                                   title: "UV Index",
//                                   value:
//                                       _weather?.getUvIndexCategory() ?? "Low",
//                                   iconPath: 'assets/icons/lucide_sun.svg',
//                                   iconColor: Colors.amber,
//                                 ),
//                                 WeatherInfo(
//                                   title: "Humidity",
//                                   value: "${_weather?.humidity ?? 0}%",
//                                   iconPath: 'assets/icons/lucide_droplets.svg',
//                                   iconColor: Colors.blue,
//                                 ),
//                                 WeatherInfo(
//                                   title: "Wind Speed",
//                                   value: "${_weather?.windSpeed ?? 0} km/h",
//                                   iconPath: 'assets/icons/lucide_wind.svg',
//                                   iconColor: Colors.blueGrey,
//                                 ),
//                               ],
//                             ),
//                             const SizedBox(height: 12),
//                             Text(
//                               'Last updated: ${DateFormat('h:mm a').format(DateTime.now())}',
//                               style: TextStyle(
//                                 fontSize: 12,
//                                 color: AppColors.textSecondary.withOpacity(0.8),
//                               ),
//                             ),
//                           ],
//                         )
//                       : const SizedBox())
//             ],
//           ),
//         ),
//       ),
//     );
//   }
// }
