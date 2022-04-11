/**
 * This is a copy of nav2_costmap_2d package
 */

#ifndef CEILING_PERCEPTION__ARRAY_PARSER_H_
#define CEILING_PERCEPTION__ARRAY_PARSER_H_


#include <vector>
#include <string>

namespace ceiling_perception
{

/** @brief Parse a vector of vectors of floats from a string.
 * @param error_return If no error, error_return is set to "".  If
 *        error, error_return will describe the error.
 * Syntax is [[1.0, 2.0], [3.3, 4.4, 5.5], ...]
 *
 * On error, error_return is set and the return value could be
 * anything, like part of a successful parse. */
std::vector<std::vector<float>> parseVVF(const std::string & input, std::string & error_return);

}  // end namespace ceiling_perception

#endif // CEILING_PERCEPTION__ARRAY_PARSER_H_
