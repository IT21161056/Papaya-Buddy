import 'dart:async';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:intl/intl.dart';
import 'package:mobile_app/services/auth_services.dart';
import 'package:mobile_app/services/plan_service.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:mobile_app/views/subscription/subscription_screen.dart';
import 'package:mobile_app/widgets/loader.widget/loading_overlay.dart';
import 'package:mobile_app/widgets/profile.widgets/accout_settings_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/contact_info_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/logout_alert_dialog.dart';
import 'package:mobile_app/widgets/profile.widgets/profile_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/subscription_widget.dart';
import 'package:mobile_app/widgets/profile.widgets/support_widget.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  _ProfileScreenState createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final AuthService _authService = AuthService();
  final PlanService _planService = PlanService();
  Map<String, dynamic>? userData;
  Map<String, dynamic>? currentPlanDetails;

  bool notificationsEnabled = true;
  bool locationEnabled = true;
  bool analyticsEnabled = true;
  double logoutScale = 1.0;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    loadUserData();
  }

  Future<void> loadUserData() async {
    userData = await _authService.getUserDetails();
    if (userData != null) {
      // Fetch the current plan details
      currentPlanDetails = await _planService
          .getPlanById(userData!['active_plan_id'] ?? AuthService.freePlanId);
    }
    setState(() => _isLoading = false);
  }

  Timer? _logoutTimeout;

  void handleLogout() async {
    bool confirmLogout = await showDialog(
      context: context,
      builder: (context) => LogoutDialog(),
    );

    if (confirmLogout == true) {
      setState(() => _isLoading = true);

      try {
        await _authService.signOut();

        if (mounted) {
          setState(() => _isLoading = false);
          Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
        }
      } catch (e) {
        if (mounted) {
          setState(() => _isLoading = false);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text("Error: ${e.toString()}")),
          );
        }
      }
    }
  }

  void animateLogout(bool isPressed) {
    setState(() {
      logoutScale = isPressed ? 0.95 : 1.0;
    });
  }

  void _updateProfilePicture(String newImageUrl) {
    if (userData != null) {
      setState(() {
        userData!['profilePicture'] = newImageUrl;
      });
    }
  }

  Future<void> _refreshData() async {
    setState(() => _isLoading = true);
    await loadUserData();
    if (mounted) {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        title: const Text(
          "Profile",
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        centerTitle: false,
        backgroundColor: Colors.white,
        elevation: 1,
        actions: [
          IconButton(
            onPressed: () {},
            icon: SvgPicture.asset(
              'assets/icons/lucide_bell.svg',
              height: 22,
              color: Colors.black,
            ),
          )
        ],
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(
              color: AppColors.primary,
            ))
          : SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  ProfileCard(
                    userName: userData?['fullName'] ?? 'User',
                    description: 'Plant enthusiast & organic farmer',
                    profilePicture: userData?['image_url'] ??
                        'https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?q=80',
                    onProfilePictureUpdated: _updateProfilePicture,
                  ),

                  const SizedBox(height: 24),
                  ContactInfoCard(
                    email: userData?['email'] ?? 'Not available',
                    phone: userData?['phoneNumber'] ?? 'Not available',
                    address: userData?['city'] ?? 'Not available',
                  ),
                  const SizedBox(height: 24),
                  AccountSettingsCard(),
                  const SizedBox(height: 24),

                  SubscriptionPlanCard(
                    currentPlan: currentPlanDetails?['name'] ?? 'Free Plan',
                    description: currentPlanDetails?['features']?.join(', ') ??
                        'Basic features with limited scans',
                    renewalDate: userData?['subscription_expiry'] != null
                        ? DateFormat('MMM dd, yyyy').format(
                            (userData!['subscription_expiry'] as Timestamp)
                                .toDate())
                        : 'Never expires',
                    plan: currentPlanDetails?['price'] == 0
                        ? 'Free'
                        : '${currentPlanDetails?['price']} LKR/${userData?['active_plan_id']?.contains('monthly') ? 'month' : 'year'}',
                    active: true,
                    onTap: () async {
                      final planUpdated = await Navigator.push<bool>(
                        context,
                        MaterialPageRoute(
                          builder: (context) => const SubscriptionScreen(),
                        ),
                      );

                      if (planUpdated == true && mounted) {
                        await _refreshData();
                      }
                    },
                  ),

                  const SizedBox(height: 24),
                  SupportCard(),

                  const SizedBox(height: 16),
                  // Logout Button
                  SizedBox(
                    width: double.infinity,
                    child: TextButton(
                      onPressed: handleLogout,
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 24),
                        backgroundColor: const Color(0xFFFEF2F2),
                        foregroundColor: const Color(0xFFEF4444),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.logout,
                            color: Colors.red,
                          ),
                          SizedBox(width: 8),
                          Text(
                            "Log Out",
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),

                  const SizedBox(height: 20),
                  const Text(
                    "PapayaBuddy v1.0.0",
                    style: TextStyle(color: Color(0xFF94A3B8)),
                  ),
                ],
              ),
            ),
      // The LoadingOverlay widget should be used in a Stack if you want an overlay,
      // but since we're already handling loading with a condition, it's redundant here
    );
  }
}
