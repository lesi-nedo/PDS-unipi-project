#include "utils.h"



/**
 * @brief Save an Marray of Labels (double) to a CSV file.
 * 
 * Writes the contents of an Marray of Labels (double) to a CSV file.
 * Each row in the Marray corresponds to a row in the CSV file, and each column corresponds to a column in the CSV file.
 * 
 * @param data The Marray of Labels (double) to be saved.
 * @param filename The name of the CSV file to be created.
 * @param writeHeader If true, the first row of the CSV file will contain the column headers (default is true).
 * 
 * * @throws std::runtime_error if there is an error opening the file for writing or if there is an error writing to the file.
 * * @throws std::invalid_argument if the Marray is empty or if the data cannot be written to the file.
 * 
 * @example
 * ```cpp
 * #include "src/utils/utils.h"
 * 
 * // Create an Marray of Labels (double)
 * andres::Marray<Label> data(3, 2); // 3 rows, 2 columns
 * data(0, 0) = 1.0; data(0, 1) = 2.0;
 * data(1, 0) = 3.0; data(1, 1) = 4.0;
 * data(2, 0) = 5.0; data(2, 1) = 6.0;
 * * // Save the Marray to a CSV file
 * 
 * 
 * 
 */

// void saveMarrayToCSV(const andres::Marray<Label>& data, const std::string_view filename, const bool writeHeader = true) {
// }