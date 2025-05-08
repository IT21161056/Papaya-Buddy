import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:mobile_app/services/auth_services.dart';
import 'package:mobile_app/services/plan_service.dart';
import 'package:mobile_app/theme/colors.dart';
import 'package:flutter_svg/flutter_svg.dart';

class SubscriptionScreen extends StatefulWidget {
  const SubscriptionScreen({super.key});

  @override
  State<SubscriptionScreen> createState() => _SubscriptionScreenState();
}

class _SubscriptionScreenState extends State<SubscriptionScreen> {
  final PlanService _planService = PlanService();
  final AuthService _authService = AuthService();
  List<Map<String, dynamic>> _plans = [];
  String? _currentPlanId;
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    try {
      final plans = await _planService.getAvailablePlans();
      final userData = await _authService.getUserDetails();

      if (mounted) {
        setState(() {
          _plans = plans;
          _currentPlanId = userData?['active_plan_id'];
          _isLoading = false;
          _error = null;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _error = 'Failed to load plans. Please try again.';
        });
      }
    }
  }

  Future<bool> _upgradePlan(String planId) async {
    if (planId == _currentPlanId) return false;

    setState(() => _isLoading = true);

    try {
      final userId = _authService.getUserUID();
      if (userId == null) throw Exception('User not logged in');

      // Update user's plan
      final expiryDate = _calculateExpiryDate(planId);
      await FirebaseFirestore.instance.collection('users').doc(userId).update({
        'active_plan_id': planId,
        'subscription_expiry': expiryDate,
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Plan upgraded successfully!')),
        );
        setState(() => _currentPlanId = planId);
      }
      return true;
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: ${e.toString()}')),
        );
      }
      return false;
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  DateTime _calculateExpiryDate(String planId) {
    return planId.contains('monthly')
        ? DateTime.now().add(const Duration(days: 30))
        : DateTime.now().add(const Duration(days: 365));
  }

  Future<void> _showPaymentConfirmation(String planId) async {
    // Replace with actual payment processing
    return showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Confirm Payment'),
        content: Text(
            'You are about to upgrade to ${_plans.firstWhere((p) => p['id'] == planId)['name']}'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              // In real app, process payment here
            },
            child: const Text('Confirm'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        title: const Text(
          "Subscription Plans",
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        centerTitle: false,
        backgroundColor: Colors.white,
        elevation: 1,
      ),
      body: _isLoading
          ? const Center(
              child: CircularProgressIndicator(color: AppColors.primary))
          : _error != null
              ? Center(child: Text(_error!))
              : _plans.isEmpty
                  ? const Center(child: Text('No plans available'))
                  : _buildPlanList(),
    );
  }

  Widget _buildPlanList() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Text(
              'Choose the right plan for your needs',
              style: TextStyle(
                fontSize: 14,
                color: Color.fromRGBO(100, 116, 139, 1),
              ),
            ),
          ),
          const SizedBox(height: 16),
          Expanded(
            child: ListView.separated(
              itemCount: _plans.length,
              separatorBuilder: (_, __) => const SizedBox(height: 16),
              itemBuilder: (context, index) => PlanCard(
                plan: _plans[index],
                isCurrent: _plans[index]['id'] == _currentPlanId,
                onUpgrade: () => _upgradePlan(_plans[index]['id']),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class PlanCard extends StatelessWidget {
  final Map<String, dynamic> plan;
  final bool isCurrent;
  final VoidCallback onUpgrade;

  const PlanCard({
    super.key,
    required this.plan,
    required this.isCurrent,
    required this.onUpgrade,
  });

  @override
  Widget build(BuildContext context) {
    final isFree = plan['price'] == 0;
    final priceText = isFree
        ? 'Free'
        : 'Rs. ${plan['price']}/${plan['id'].contains('monthly') ? 'month' : 'year'}';

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
        border: Border.all(
          color: isCurrent ? AppColors.primary : Colors.transparent,
          width: 1.5,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                plan['name'],
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Color.fromRGBO(26, 26, 26, 1),
                ),
              ),
              const Spacer(),
              if (isCurrent)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    'Active',
                    style: TextStyle(
                      color: AppColors.primary,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            priceText,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Color.fromRGBO(26, 26, 26, 1),
            ),
          ),
          const SizedBox(height: 16),
          ...(plan['features'] as List).map((feature) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SvgPicture.asset(
                      'assets/icons/lucide_check.svg',
                      height: 18,
                      width: 18,
                      color: AppColors.primary,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        feature.toString(),
                        style: const TextStyle(
                          fontSize: 14,
                          color: Color.fromRGBO(26, 26, 26, 1),
                        ),
                      ),
                    ),
                  ],
                ),
              )),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity,
            child: TextButton(
              onPressed: isCurrent ? null : onUpgrade,
              style: TextButton.styleFrom(
                backgroundColor:
                    isCurrent ? const Color(0xFFF1F5F9) : AppColors.primary,
                foregroundColor:
                    isCurrent ? const Color(0xFF94A3B8) : Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Text(
                isCurrent ? 'Current Plan' : 'Upgrade Now',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
