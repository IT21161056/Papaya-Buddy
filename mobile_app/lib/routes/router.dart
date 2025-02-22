import 'package:flutter/material.dart';
import '../views/home_view.dart';
import '../views/splash_view.dart';
import '../views/auth/login_view.dart';

class AppRouter {
  static Route<dynamic> generateRoute(RouteSettings settings) {
    switch (settings.name) {
      case '/':
        return MaterialPageRoute(builder: (_) => const SplashView());
      case '/home':
        return MaterialPageRoute(builder: (_) => const HomeView());
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
