#include <CL/sycl.hpp>

#include <iostream>
#include <optional>
#include <unordered_set>
#include <vector>

#include <glm/ext/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

using namespace cl::sycl;

typedef int16_t coord_t;
typedef std::vector<glm::vec4> scanner_t;

#define ELEMS(x) (sizeof(x) / sizeof(x[0]))

static bool in_range(glm::vec4 vec) {
	return vec[0] <= 1000 && vec[1] <= 1000 && vec[2] <= 1000 &&
	       vec[0] >= -1000 && vec[1] >= -1000 && vec[2] >= -1000;
}

struct hashFunction {
	size_t operator()(const glm::vec4 &x) const {
		return static_cast<size_t>(x[0]) ^ static_cast<size_t>(x[1]) ^
		       static_cast<size_t>(x[2]) ^ static_cast<size_t>(x[3]);
	}
};

static std::optional<glm::mat4> fit_scanner(
	const scanner_t &scanner1,
	const scanner_t &scanner2,
	const std::vector<glm::mat4> &orientation_matrices
) {
	queue q;

	buffer<glm::vec4, 1> buf_scanner1(range<1>{scanner1.size()});
	{
		auto scanner1_w = buf_scanner1.get_access<access::mode::write>();
		for (size_t i = 0; i < scanner1.size(); i++) {
			scanner1_w[i] = scanner1[i];
		}
	}

	buffer<glm::vec4, 1> buf_scanner2(range<1>{scanner2.size()});
	{
		auto scanner2_w = buf_scanner2.get_access<access::mode::write>();
		for (size_t i = 0; i < scanner2.size(); i++) {
			scanner2_w[i] = scanner2[i];
		}
	}

	buffer<glm::mat4, 1> buf_orientation_matrices(range<1>{orientation_matrices.size()});
	{
		auto orientation_matrices_w = buf_orientation_matrices.get_access<access::mode::write>();
		for (size_t i = 0; i < orientation_matrices.size(); i++) {
			orientation_matrices_w[i] = orientation_matrices[i];
		}
	}

	buffer<glm::vec4, 2> rotated_scanner2(range<2>{orientation_matrices.size(), scanner2.size()});

	q.submit([&](handler &cgh) {
		auto scanner2_r = buf_scanner2.get_access<access::mode::read>(cgh);
		auto orientation_matrices_r = buf_orientation_matrices.get_access<access::mode::read>(cgh);
		auto rotated_scanner2_w = rotated_scanner2.get_access<access::mode::write>(cgh);

		cgh.parallel_for(range<2>{
			orientation_matrices.size(),
			scanner2.size(),
		}, [=](id<2> index) {
			rotated_scanner2_w[index] = orientation_matrices_r[index[0]] * scanner2_r[index[1]];
		});
	});

	auto scanner1_size = scanner1.size();
	auto scanner2_size = scanner2.size();
	buffer<bool, 2> buf_scanner1_covered(range<2>{
		scanner1.size() * orientation_matrices.size() * scanner2.size(),
		scanner1.size(),
	});

	buffer<bool, 3> buf_match(range<3>{
		scanner1.size(),
		orientation_matrices.size(),
		scanner2.size(),
	});

	q.submit([&](handler &cgh) {
		auto scanner1_r = buf_scanner1.get_access<access::mode::read>(cgh);
		auto rotated_scanner2_r = rotated_scanner2.get_access<access::mode::read>(cgh);
		auto scanner1_covered_rw = buf_scanner1_covered.get_access<access::mode::read_write>(cgh);
		auto match_w = buf_match.get_access<access::mode::write>(cgh);

		cgh.parallel_for(range<3>{
			scanner1.size(),
			orientation_matrices.size(),
			scanner2.size(),
		}, [=](id<3> index) {
			size_t beacon1_index = index[0];
			size_t orientation_index = index[1];
			size_t beacon2_index = index[2];

			auto covered = scanner1_covered_rw[
				orientation_index * scanner1_size * scanner2_size +
				beacon1_index * scanner2_size + beacon2_index
			];

			const auto &beacon1 = scanner1_r[beacon1_index];
			const auto &beacon2 = rotated_scanner2_r[orientation_index][beacon2_index];
			auto diff = beacon1 - beacon2;
			auto scanner2_center = glm::vec4 {0, 0, 0, 1} + diff;

			for (size_t i = 0; i < scanner1_size; i++) {
				covered[i] = false;
			}

			int overlaps = 0;
			bool mismatch = false;
			for (size_t i = 0; i < scanner2_size; i++) {
				auto translated_beacon = rotated_scanner2_r[orientation_index][i] + diff;

				bool found_match = false;
				for (size_t j = 0; j < scanner1_size; j++) {
					auto curr_beacon1 = scanner1_r[j];

					if (curr_beacon1 == translated_beacon) {
						found_match = true;
						covered[j] = true;
						break;
					}
				}
				if (found_match) {
					overlaps++;
				} else if (in_range(translated_beacon)) {
					mismatch = true;
					break;
				}
			}

			for (size_t i = 0; i < scanner1_size; i++) {
				if (in_range(scanner1_r[i] - diff) && !covered[i]) {
					mismatch = true;
					break;
				}
			}

			match_w[index] = overlaps >= 12 && !mismatch;
		});
	});

	auto match_r = buf_match.get_access<access::mode::read>();
	auto rotated_scanner2_r = rotated_scanner2.get_access<access::mode::read>();
	for (size_t i = 0; i < scanner1_size; i++) {
		for (size_t j = 0; j < orientation_matrices.size(); j++) {
			for (size_t k = 0; k < scanner2_size; k++) {
				if (match_r[i][j][k]) {
					auto diff = scanner1[i] - rotated_scanner2_r[j][k];
					return glm::translate(glm::mat4(1), glm::vec3(diff[0], diff[1], diff[2])) * orientation_matrices[j];
				}
			}
		}
	}

	return {};
}

int main() {
	std::vector<scanner_t> scanners;

	std::string line;
	while (std::getline(std::cin, line)) {
		coord_t a;
		coord_t b;
		coord_t c;
		scanner_t scanner;

		while (scanf("%" SCNi16 ",%" SCNi16 ",%" SCNi16, &a, &b, &c) == 3) {
			scanner.emplace_back(a, b, c, 1);
		}

		scanners.push_back(std::move(scanner));
	}

	std::vector<std::optional<glm::mat4>> scanner_orientations(scanners.size());

	// Identity matrix, as orientations will be in scanners[0]'s frame of reference
	scanner_orientations.at(0) = glm::mat4{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	};

	// Each possible facing of the x-axis
	glm::mat4 facing_matrices[] = {
		{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		},
		{
			{-1, 0, 0, 0},
			{0, -1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		},
		{
			{0, -1, 0, 0},
			{1, 0, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		},
		{
			{0, 1, 0, 0},
			{-1, 0, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		},
		{
			{0, 0, -1, 0},
			{0, 1, 0, 0},
			{1, 0, 0, 0},
			{0, 0, 0, 1},
		},
		{
			{0, 0, 1, 0},
			{0, 1, 0, 0},
			{-1, 0, 0, 0},
			{0, 0, 0, 1},
		},
	};

	// Each possible rotation around the x-axis
	glm::mat4 rotation_matrices[] = {
		{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1},
		},
		{
			{1, 0, 0, 0},
			{0, -1, 0, 0},
			{0, 0, -1, 0},
			{0, 0, 0, 1},
		},
		{
			{1, 0, 0, 0},
			{0, 0, 1, 0},
			{0, -1, 0, 0},
			{0, 0, 0, 1},
		},
		{
			{1, 0, 0, 0},
			{0, 0, -1, 0},
			{0, 1, 0, 0},
			{0, 0, 0, 1},
		},
	};

	// Each possible (non-mirrored) orientation
	std::vector<glm::mat4> orientation_matrices(ELEMS(facing_matrices) * ELEMS(rotation_matrices));

	for (size_t i = 0; i < ELEMS(facing_matrices); i++) {
		for (size_t j = 0; j < ELEMS(rotation_matrices); j++) {
			orientation_matrices.at(i + j * ELEMS(facing_matrices)) = facing_matrices[i] * rotation_matrices[j];
		}
	}

	std::vector<size_t> fitted_scanners;
	fitted_scanners.push_back(0);

	for (size_t i = 0; i < scanners.size(); i++) {
		auto n = fitted_scanners.at(i);
		const auto &scanner1 = scanners.at(n);

		for (size_t m = 0; m < scanners.size(); m++) {
			if (m == n || scanner_orientations.at(m)) {
				continue;
			}

			auto orientation = fit_scanner(scanner1, scanners.at(m), orientation_matrices);

			if (orientation.has_value()) {
				scanner_orientations[m] = scanner_orientations[n].value() * orientation.value();
				fitted_scanners.push_back(m);
			}
		}
	}

	std::unordered_set<glm::vec4, hashFunction> beacons;

	for (size_t i = 0; i < scanners.size(); i++) {
		const auto &scanner = scanners.at(i);
		const auto &orientation = scanner_orientations.at(i).value();
		for (const auto &beacon: scanner) {
			beacons.insert(orientation * beacon);
		}
	}

	std::cout << beacons.size() << std::endl;
}
