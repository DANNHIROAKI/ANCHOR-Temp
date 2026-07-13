CXX ?= c++
BUILD_DIR ?= build

CPPFLAGS += -Icpp/include -Icpp/app
WARNINGS := -Wall -Wextra -Wconversion -Wshadow
CXXFLAGS ?= -O3 -DNDEBUG
CXXFLAGS += -std=c++20 $(WARNINGS) -MMD -MP

CORE_OBJECT := $(BUILD_DIR)/obj/cpp/src/anchor.o
APP_OBJECTS := \
	$(BUILD_DIR)/obj/cpp/app/benchmark_main.o \
	$(BUILD_DIR)/obj/cpp/app/workload.o \
	$(BUILD_DIR)/obj/cpp/app/sha256.o
TEST_OBJECT := $(BUILD_DIR)/obj/cpp/tests/test_core.o
OBJECTS := $(CORE_OBJECT) $(APP_OBJECTS) $(TEST_OBJECT)

.PHONY: all release test clean

all: release

release: $(BUILD_DIR)/anchor_bench

test: $(BUILD_DIR)/anchor_core_tests
	$(BUILD_DIR)/anchor_core_tests

$(BUILD_DIR)/anchor_bench: $(CORE_OBJECT) $(APP_OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/anchor_core_tests: $(CORE_OBJECT) $(TEST_OBJECT)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/obj/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

-include $(OBJECTS:.o=.d)
