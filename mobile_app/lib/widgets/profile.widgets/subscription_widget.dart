import 'package:flutter/material.dart';
import 'package:mobile_app/theme/colors.dart';

class SubscriptionPlanCard extends StatelessWidget {
  final String currentPlan;
  final bool active;
  final String renewalDate;
  final String plan;
  final VoidCallback onTap;
  final String description;

  const SubscriptionPlanCard({
    super.key,
    required this.currentPlan,
    required this.active,
    required this.renewalDate,
    required this.plan,
    required this.onTap,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    bool isFreePlan = plan.toLowerCase().contains('free');

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isFreePlan
              ? Colors.grey.withOpacity(0.3)
              : AppColors.primary.withOpacity(0.3),
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                currentPlan,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                  color: isFreePlan ? Colors.grey[800] : AppColors.primary,
                ),
              ),
              const Spacer(),
              if (active)
                Container(
                  padding:
                      const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    color: isFreePlan
                        ? Colors.grey.withOpacity(0.2)
                        : AppColors.primary.withOpacity(0.2),
                  ),
                  child: Text(
                    "Active",
                    style: TextStyle(
                      color: isFreePlan ? Colors.grey : AppColors.primary,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            description,
            style: TextStyle(
              fontSize: 14,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppColors.cardForeground,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              children: [
                if (!isFreePlan) _buildDetailRow("Renewal Date", renewalDate),
                _buildDetailRow(isFreePlan ? "Plan Type" : "Billing", plan),
                if (!isFreePlan) const SizedBox(height: 8),
                if (!isFreePlan)
                  _buildDetailRow(
                    "Scans Available",
                    currentPlan.contains('Monthly') ? "100/month" : "Unlimited",
                  ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: double.infinity,
            child: TextButton(
              onPressed: onTap,
              style: TextButton.styleFrom(
                backgroundColor: isFreePlan
                    ? Colors.blue.withOpacity(0.1)
                    : AppColors.primary.withOpacity(0.1),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 12),
                child: Text(
                  isFreePlan ? "Upgrade Plan" : "Manage Subscription",
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: isFreePlan ? Colors.blue : AppColors.primary,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

Widget _buildDetailRow(String label, String value) {
  return Row(
    mainAxisAlignment: MainAxisAlignment.spaceBetween,
    children: [
      Text(
        label,
        style: const TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w500,
          color: AppColors.textSecondary,
        ),
      ),
      Text(
        value,
        style: const TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: Colors.black,
        ),
      ),
    ],
  );
}
