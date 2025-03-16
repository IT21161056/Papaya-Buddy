import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'dart:async';

import 'package:mobile_app/views/home/home_view.dart';

class SplashView extends StatefulWidget {
  const SplashView({super.key});

  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashView> {
  double progressWidth = 0;

  @override
  void initState() {
    super.initState();

    Future.delayed(const Duration(milliseconds: 500), () {
      setState(() {
        progressWidth = 200;
      });
    });

    Future.delayed(const Duration(seconds: 2), () {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const HomePage()),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  width: 150,
                  height: 150,
                  decoration: BoxDecoration(
                    // color: Colors.green.shade100,
                    shape: BoxShape.circle,
                    // boxShadow: [
                    //   BoxShadow(
                    //     color: Colors.black.withOpacity(0.1),
                    //     blurRadius: 12,
                    //     offset: const Offset(0, 4),
                    //   ),
                    // ],
                  ),
                  child: Center(
                    child: Image.asset(
                      'assets/app_icon_3.png', // Replace with actual asset path
                      // height: 100,
                      // width: 100,
                      // color:
                      // Colors.green, // If you want to apply a color overlay
                      // colorBlendMode: BlendMode.srcIn, // Apply the blend mode
                    ),
                  ),
                )
                    .animate()
                    .scale(duration: 500.ms)
                    .fadeIn(duration: 500.ms)
                    .moveY(begin: 20, end: 0, duration: 500.ms),

                const SizedBox(height: 12), // Spacing

                const Text(
                  "PapayaBuddy",
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                  ),
                )
                    .animate()
                    .fadeIn(duration: 600.ms)
                    .moveY(begin: 20, end: 0, duration: 600.ms),

                const SizedBox(height: 8),

                const Text(
                  "Your plant's health companion",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.grey,
                  ),
                )
                    .animate()
                    .fadeIn(duration: 700.ms)
                    .moveY(begin: 20, end: 0, duration: 700.ms),

                const SizedBox(height: 80),

                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 40.0),
                  child: Container(
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.grey.shade300,
                      borderRadius: BorderRadius.circular(2),
                    ),
                    child: Stack(
                      children: [
                        AnimatedContainer(
                          duration: const Duration(milliseconds: 1500),
                          curve: Curves.easeInOut,
                          width: progressWidth,
                          height: 4,
                          decoration: BoxDecoration(
                            color: Colors.green,
                            borderRadius: BorderRadius.circular(2),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 50),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
