import 'package:flutter/material.dart';
import 'package:mobile_app/views/auth/signup_view.dart';
import 'package:mobile_app/views/home/home_view.dart';
import '../views/splash_view.dart';
import '../views/auth/login_view.dart';

class AppRouter {
  static Route<dynamic> generateRoute(RouteSettings settings) {
    switch (settings.name) {
      case '/':
        return MaterialPageRoute(builder: (_) => const SplashView());
      case '/home':
        return MaterialPageRoute(builder: (_) => const HomePage());
      case '/signup':
        return MaterialPageRoute(builder: (_) => const SignupView());
      case '/login':
        return MaterialPageRoute(
            builder: (_) => const LoginView()); // Added login route
      default:
        return MaterialPageRoute(
          builder: (_) => const Scaffold(
            body: Center(child: Text('Page Not Found')),
          ),
        );
    }
  }
}
