import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:mobile_app/models/predictionModel.dart';
import 'package:mobile_app/services/predictionService.dart';

class PredictionsList extends StatefulWidget {
  final String userId;
  final double height;
  final Function(bool)? onLoadingChanged;

  const PredictionsList({
    Key? key,
    required this.userId,
    this.height = 300,
    this.onLoadingChanged,
  }) : super(key: key);

  @override
  _PredictionsListState createState() => _PredictionsListState();
}

class _PredictionsListState extends State<PredictionsList> {
  List<Prediction> predictions = [];
  bool isLoading = true;
  String errorMessage = '';

  @override
  void initState() {
    super.initState();
    _loadPredictions();
  }

  Future<void> _loadPredictions() async {
    setState(() {
      isLoading = true;
      errorMessage = '';
    });

    widget.onLoadingChanged?.call(true);

    try {
      List<Prediction> fetchedPredictions = await HistoryService.getDiagnosis(
        userId: widget.userId,
      );

      if (mounted) {
        setState(() {
          predictions = fetchedPredictions;
          isLoading = false;
        });
        widget.onLoadingChanged?.call(false);
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          errorMessage = 'Failed to load predictions';
          isLoading = false;
        });
        widget.onLoadingChanged?.call(false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: widget.height,
      child: _buildContent(),
    );
  }

  Widget _buildContent() {
    if (isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (errorMessage.isNotEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(errorMessage, style: TextStyle(color: Colors.red[700])),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadPredictions,
              child: const Text('Try Again'),
            ),
          ],
        ),
      );
    }

    if (predictions.isEmpty) {
      return const Center(
        child: Text(
          'No diagnoses found.\nGet started by scanning your first plant!',
          textAlign: TextAlign.center,
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadPredictions,
      child: ListView.separated(
        padding: EdgeInsets.zero,
        itemCount: predictions.length,
        separatorBuilder: (_, __) => const SizedBox(height: 10),
        itemBuilder: (context, index) {
          final prediction = predictions[index];
          return _buildPredictionListItem(prediction);
        },
      ),
    );
  }

  Widget _buildPredictionListItem(Prediction prediction) {
    final dateFormat = DateFormat('MMM d, yyyy');
    final formattedDate =
        dateFormat.format(DateTime.parse(prediction.createdAt));

    final diseaseName = prediction.disease?.name ?? 'Unknown';
    final isHealthy = diseaseName.toLowerCase().contains('healthy');
    final resultStatus = isHealthy ? 'Healthy' : 'Unhealthy';

    return DiagnosisListItem(
      title: diseaseName,
      date: formattedDate,
      result: resultStatus,
      onDetailsPressed: () {
        // Handle details action
      },
    );
  }
}

class DiagnosisListItem extends StatelessWidget {
  final String title;
  final String date;
  final String result;
  final VoidCallback onDetailsPressed;

  const DiagnosisListItem({
    Key? key,
    required this.title,
    required this.date,
    required this.result,
    required this.onDetailsPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 5,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  date,
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 4),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color:
                        result == 'Healthy' ? Colors.green[50] : Colors.red[50],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    result,
                    style: TextStyle(
                      color: result == 'Healthy' ? Colors.green : Colors.red,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
          ),
          TextButton(
            onPressed: onDetailsPressed,
            child: const Text('Details'),
          ),
        ],
      ),
    );
  }
}
