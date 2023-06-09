#include <math.h>

// In a real implementation, the following data are downloaded from the coordinator at the beginning. 
float threshold = 2.12093; 
float baseline_mean[6] = {0.01736804, 0.02250047, 0.00201864, 0.00190659, 0.08863717, 0.00134513};
double baseline_cov_inv[6][6] = {
                            {6.58420124e+02, -2.88850322e+02, 2.96812078e+00, -1.17943892e+01, 3.91937593e+00, 3.53805669e+00},
                            {-2.88850322e+02, 5.53155945e+02, 2.44273747e+00, 7.36841971e+00, 3.04711095e+00, 3.87166757e+00},
                            {2.96812078e+00, 2.44273747e+00, 4.46926579e+02, -3.39201122e+00, -6.02202348e-01, -1.71135799e+00},
                            {-1.17943892e+01, 7.36841971e+00, -3.39201122e+00, 5.49853896e+02, -5.20381496e+00, 6.35956445e-01},
                            {3.91937593e+00, 3.04711095e+00, -6.02202348e-01, -5.20381496e+00, 2.60455788e+01, 2.95433572e-02},
                            {3.53805669e+00, 3.87166757e+00, -1.71135799e+00, 6.35956445e-01, 2.95433572e-02, 8.99834810e+02}
                            };
float anomalous_mean[6] = {0.27989781, 0.07662126, 0.21563614, 0.11193405, 0.18611995, 0.49844202}; 
float anomalous_cov_inv[6][6] = {{31.93769273, -31.12636613, -7.48273897, -6.56859793, 2.26676152, -9.62129698},
                           {-31.12636613, 75.24293555, 17.83533311, 13.10716607, 2.66858428, 21.08088637},
                           {-7.48273897, 17.83533311, 13.78783272, 1.80527277, 4.1954201, 10.27753895},
                           {-6.56859793, 13.10716607, 1.80527277, 22.03408234, 2.29798367, 5.23939852},
                           {2.26676152, 2.66858428, 4.1954201, 2.29798367, 12.11176229, 6.25559921},
                           {-9.62129698, 21.08088637, 10.27753895, 5.23939852, 6.25559921, 14.3421671}};


// ground truth 
int true_pred[] = {1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1};

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  // 100 data points with 6 features from the NSL-KDD dataset 
  float data[100][6] = {
  {0.00195695, 0.00195695, 0, 0, 0.99, 0},
  {0.00195695, 0.00195695, 0, 0, 0.01, 0},
  {0.00391389, 0.00391389, 0, 0, 0, 0},
  {0.00391389, 0.00391389, 0, 0, 0.28, 0},
  {0.00195695, 0.00195695, 1, 0, 0.02, 0},
  {0.01369863, 0.02544031, 0, 0, 0, 0},
  {0.00391389, 0.00195695, 0, 1, 0.49, 0},
  {0.22700587, 0.03131115, 0, 0.06, 0, 1},
  {0.37377691, 0.01956947, 1, 0.07, 0, 0},
  {0.54794521, 0.00587084, 0, 0.05, 0, 1},
  {0.03522505, 0.03522505, 0, 0, 0.06, 0},
  {0.31506849, 0.01369863, 0, 0.06, 0, 1},
  {0.47162427, 0.03131115, 1, 0.06, 0, 0},
  {0.00587084, 0.01956947, 0, 0, 0.02, 0.02},
  {0.47553816, 0.02348337, 0, 0.06, 0, 1},
  {0.00195695, 0.00195695, 0, 0, 0.01, 0},
  {0.00587084, 0.00587084, 0, 0, 0, 0},
  {0.00195695, 0.00195695, 1, 0, 0.24, 0},
  {0.00978474, 0.01565558, 0, 0, 0, 0},
  {0.53424658, 0.0332681, 0, 0.06, 0, 1},
  {0.00195695, 0.06457926, 0, 0, 1, 0},
  {0.63405088, 0.00195695, 1, 0.51, 1, 0},
  {0.02348337, 0.00391389, 0, 0.33, 0.01, 1},
  {0.8297456, 0.8297456, 0, 0, 1, 0},
  {0.00195695, 0.00195695, 0, 0, 0.01, 0},
  {0.06457926, 0.01369863, 0, 0.12, 0.01, 0.96},
 {0.06262231, 0.07240705, 0, 0, 0, 0},
 {0.00978474, 0.00978474, 0, 0, 0.2, 0},
 {0.1409002, 0.04109589, 0, 0.06, 0, 1},
 {0.02935421, 0.02935421, 0, 0, 0.06, 0},
 {0.20156556, 0.0332681, 0, 0.06, 0, 1},
 {0.01369863, 0.01369863, 0, 0, 0.01, 0},
 {0.01369863, 0.01369863, 0, 0, 1, 0},
 {0.00195695, 0.00195695, 0, 0, 0, 0.88},
 {0.00391389, 0.00391389, 0, 0, 0.05, 0},
 {0.02152642, 0.00782779, 0, 0.27, 0.66, 0},
 {0.47749511, 0.02739726, 0, 0.06, 0, 1},
 {0.22896282, 0.037182, 0, 0.06, 0, 1},
 {0.00195695, 0.00195695, 0, 0, 0, 0},
 {0.04305284, 0.04305284, 0, 0, 0, 0},
 {0.14872798, 0.30528376, 0, 0.03, 0, 0},
 {0.00391389, 0.00195695, 0, 1, 0, 0.09},
 {0.38747554, 0.00195695, 0, 0.06, 0, 1},
 {0.54990215, 0.03131115, 1, 0.06, 0, 0},
 {0.09197652, 0.01565558, 0, 0.09, 0, 1},
 {0.01174168, 0.01174168, 0, 0, 0.01, 0},
 {0.39530333, 0.02348337, 0, 0.06, 0, 1},
 {0.00978474, 0.01174168, 0, 0, 0.01, 0},
 {0.02152642, 0.02152642, 0, 0, 0.22, 0},
 {0.037182, 0.037182, 0, 0, 0, 0},
 {0.43248532, 0.00391389, 0.0, 0.06, 0.0, 1.0},
  {0.00782779, 0.00782779, 0.0, 0.0, 0.01, 0.0},
  {0.00782779, 0.00782779, 0.0, 0.0, 0.04, 0.0},
  {0.55577299, 0.03913894, 0.0, 0.04, 0.01, 1.0},
  {0.49902153, 0.01174168, 0.0, 0.06, 0.0, 1.0},
  {1.0, 0.00195695, 1.0, 1.0, 0.0, 0.0},
  {0.00195695, 0.00195695, 0.0, 0.0, 1.0, 0.0},
  {0.00195695, 0.00195695, 0.0, 0.0, 0.0, 0.0},
  {0.00195695, 0.00587084, 0.67, 0.0, 0.01, 0.0},
  {0.41878669, 0.00587084, 1.0, 0.07, 0.0, 0.0},
  {0.46183953, 0.03522505, 0.0, 0.06, 0.0, 1.0},
  {0.49902153, 0.00978474, 1.0, 0.07, 0.0, 0.0},
  {0.3816047, 0.3816047, 0.0, 0.0, 0.0, 0.0},
  {0.00195695, 0.00195695, 0.0, 0.0, 0.01, 0.01},
  {0.00391389, 0.00391389, 0.0, 0.0, 0.05, 0.0},
  {0.25244618, 0.03522505, 0.0, 0.06, 0.0, 1.0},
  {0.00195695, 0.00195695, 0.0, 0.0, 0.02, 0.0},
  {0.2407045, 0.04696673, 0.0, 0.05, 0.0, 1.0},
  {0.50880626, 0.03913894, 0.0, 0.06, 0.0, 1.0},
  {0.06262231, 0.06262231, 0.0, 0.0, 0.0, 0.0},
  {0.037182, 0.00195695, 0.0, 0.16, 0.09, 0.0},
  {0.02544031, 0.02544031, 0.0, 0.0, 0.0, 0.0},
  {0.02935421, 0.02935421, 0.0, 0.0, 0.07, 0.0},
  {0.03913894, 0.03913894, 0.0, 0.0, 0.73, 0.01},
  {0.02544031, 0.02544031, 0.0, 0.0, 0.02, 0.0},
    {0.33659491, 0.02544031, 0.0, 0.08, 0.0, 1.0},
  {0.51076321, 0.00587084, 1.0, 0.07, 0.0, 0.0},
  {0.06849315, 0.06849315, 0.0, 0.0, 0.0, 0.0},
  {0.02935421, 0.02935421, 0.0, 0.0, 0.0, 0.01},
  {0.00195695, 0.00195695, 0.0, 0.0, 0.0, 0.0},
  {0.42074364, 0.00978474, 0.0, 0.07, 0.0, 1.0},
  {0.037182, 0.037182, 0.0, 0.0, 0.08, 0.0},
  {0.00195695, 0.00195695, 0.0, 0.0, 0.5, 0.01},
  {0.00782779, 0.00782779, 0.0, 0.0, 0.0, 0.0},
  {0.99804305, 0.99804305, 0.0, 0.0, 0.85, 0.0},
  {0.12133072, 0.02544031, 0.0, 0.06, 0.02, 1.0},
  {0.94129159, 0.04500978, 0.83, 0.95, 0.0, 0.18},
  {0.37181996, 0.01369863, 0.0, 0.07, 0.0, 1.0},
  {0.01174168, 0.01174168, 0.0, 0.0, 0.01, 0.0},
  {0.01174168, 0.01174168, 0.0, 0.0, 0.18, 0.0},
  {0.02152642, 0.08023483, 0.0, 0.0, 0.09, 0.0},
  {0.54794521, 0.03522505, 0.0, 0.06, 0.0, 1.0},
  {0.58317025, 0.01369863, 0.0, 0.06, 0.0, 1.0},
  {0.00391389, 0.00391389, 0.0, 0.0, 0.05, 0.0},
  {0.518591, 0.037182, 0.0, 0.06, 0.0, 1.0},
  {0.01369863, 0.01369863, 0.0, 0.0, 0.03, 0.0},
  {0.09589041, 0.02544031, 0.0, 0.08, 0.0, 1.0},
  {0.5518591, 0.02348337, 0.0, 0.06, 0.0, 1.0},
  {0.5146771, 0.03522505, 0.0, 0.06, 0.0, 1.0},
  {0.00195695, 0.00195695, 1.0, 0.0, 1.0, 0.0}};


  // loop through each row in data array
  unsigned long start_time = millis(); // Get the current time
  int ids_array[100] = {};
  for (int i = 0; i < 100; i++) {
    float point[6] = {data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5]}; // get the current point as a 1D array
    float distance = mahalanobis_distance(point);
    if (distance >= threshold) {
      Serial.print("anomaly");
      ids_array[i] = 1;
    }else{
      Serial.print("Benign");
      ids_array[i] = 0;
    }
    Serial.print(" , ");
  }
  Serial.println("");
  unsigned long end_time = millis(); // Get the end time
  unsigned long elapsed_time = end_time - start_time; // Calculate the elapsed time
  Serial.println("Elapsed time: " + String(elapsed_time) + " ms"); // Print the elapsed time to the serial monitor
  
  Serial.println("*********");

  Serial.println("");
  int num_correct = 0;
  int total = sizeof(true_pred) / sizeof(true_pred[0]);
  for (int i = 0; i < total; i++) {
    if (true_pred[i] == ids_array[i]) {
      num_correct++;
    }
  }
  float accuracy = (float) num_correct / total * 100;
  Serial.print("Accuracy: ");
  Serial.print(accuracy);
  Serial.println("%");
  Serial.println("");
  delay(5000);
}



// Define a function to calculate the Mahalanobis distance for the baseline Kmean
float mahalanobis_distance(float point[]) {
  // number of features = 6
  float diff[6];
  float result[6];
  float sum = 0.0;
  // Calculate the difference between the point and the baseline mean
  for (int i = 0; i < 6; i++) {
    diff[i] = point[i] - baseline_mean[i];
  }
  // Calculate the Mahalanobis distance
  for (int i = 0; i < 6; i++) {
    result[i] = 0.0;
    for (int j = 0; j < 6; j++) {
      result[i] += diff[j] * baseline_cov_inv[j][i];
    }
    sum += result[i] * diff[i];
  }
  return sqrt(sum);
}
