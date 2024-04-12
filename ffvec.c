#include <Python.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <math.h>

#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define VEC_SIZE 384
#define QUANTIZED_VEC_SIZE 48

typedef struct
{
    PyObject *key;
    PyObject *value;
    size_t *indices;
    size_t size;
    size_t capacity;
} MetadataIndex;

typedef struct
{
    PyObject_HEAD unsigned char **vectors;
    PyObject **metadata;
    MetadataIndex *index;
    size_t index_size;
    size_t capacity;
    size_t size;
} VectorSet;

static float sigmoid(float z)
{
    return 1.0 / (1.0 + exp(-z));
}

static float cblas_ddot(const size_t n, const float *x, const size_t incx, const float *y, const size_t incy)
{
    float result = 0.0;
    size_t i = 0;

#if defined(__AVX__)
    for (; i + 8 <= n; i += 8)
    {
        __m256 x_vec = _mm256_loadu_ps(x + i * incx);
        __m256 y_vec = _mm256_loadu_ps(y + i * incy);
        __m256 prod = _mm256_mul_ps(x_vec, y_vec);
        result += prod[0] + prod[1] + prod[2] + prod[3] + prod[4] + prod[5] + prod[6] + prod[7];
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t x_vec = vld1q_f32(x + i * incx);
        float32x4_t y_vec = vld1q_f32(y + i * incy);
        float32x4_t prod = vmulq_f32(x_vec, y_vec);
        result += prod[0] + prod[1] + prod[2] + prod[3];
    }
#endif

    for (; i < n; i++)
    {
        result += x[i * incx] * y[i * incy];
    }

    return result;
}

static int hamming_distance(const unsigned char *a, const unsigned char *b, const size_t len)
{
    int total_count = 0;
    size_t i = 0;

#if defined(__AVX__)
    for (; i + 32 <= len; i += 32)
    {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i xor = _mm256_xor_si256(va, vb);
        __m256i cnt = _mm256_sad_epu8(xor, _mm256_setzero_si256());
        total_count += _mm256_extract_epi64(cnt, 0) + _mm256_extract_epi64(cnt, 2);
    }
#elif defined(__ARM_NEON)
    for (; i + 16 <= len; i += 16)
    {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t xor = veorq_u8(va, vb);
        uint8x16_t cnt = vcntq_u8(xor);
        total_count += vaddvq_u8(cnt);
    }
#endif

    for (; i < len; i++)
    {
        unsigned char x = a[i];
        unsigned char y = b[i];
        unsigned char xor = x ^ y;
        unsigned char count = 0;
        for (size_t j = 0; j < 8; j++)
        {
            count += (xor >> j) & 1;
        }
        total_count += count;
    }

    return total_count;
}

static void sgd_update(float *weights, float *input, float y, size_t feature_len, float learning_rate)
{
    float prediction = sigmoid(cblas_ddot(feature_len, weights, 1, input, 1));
    float error = prediction - y;
    for (size_t i = 0; i < feature_len; i++)
    {
        weights[i] -= learning_rate * error * input[i];
    }
}

static void metadata_index_init(MetadataIndex *index)
{
    index->key = NULL;
    index->value = NULL;
    index->indices = NULL;
    index->size = 0;
    index->capacity = 0;
}

static void metadata_index_dealloc(MetadataIndex *index)
{
    Py_XDECREF(index->key);
    Py_XDECREF(index->value);
    free(index->indices);
}

static void metadata_index_add(MetadataIndex *index, PyObject *key, PyObject *value, size_t vector_index)
{
    if (index->size == index->capacity)
    {
        size_t new_capacity = (index->capacity == 0) ? 1 : index->capacity * 2;
        size_t *new_indices = (size_t *)realloc(index->indices, new_capacity * sizeof(size_t));
        if (!new_indices)
        {
            return; // Error handling
        }
        index->indices = new_indices;
        index->capacity = new_capacity;
    }

    if (index->key == NULL)
    {
        Py_INCREF(key);
        index->key = key;
    }

    if (index->value == NULL)
    {
        Py_INCREF(value);
        index->value = value;
    }

    index->indices[index->size] = vector_index;
    index->size++;
}

static void quantize(const float *vec, size_t len, unsigned char *result)
{
    size_t num_chunks = len / 8;
    for (size_t i = 0; i < num_chunks; i++)
    {
        unsigned char chunk = 0;
        for (size_t j = 0; j < 8; j++)
        {
            uint32_t *int_view = (uint32_t *)&vec[i * 8 + j];
            // Extract the sign bit (leftmost bit), 0 for positive, 1 for negative
            unsigned char bit = (*int_view >> 31) & 1;
            chunk |= (bit << j);
        }
        result[i] = chunk;
    }
}

static void dequantize(const unsigned char *vec, size_t len, float *result)
{
    size_t num_chunks = len / 8;
    for (size_t i = 0; i < num_chunks; i++)
    {
        for (size_t j = 0; j < 8; j++)
        {
            result[i * 8 + j] = (vec[i] & (1 << j)) > 0 ? 1.0 : -1.0;
        }
    }
}

static int ffvec_init(VectorSet *self)
{
    self->capacity = 10;
    self->size = 0;
    self->vectors = (unsigned char **)malloc(self->capacity * sizeof(unsigned char *));
    if (!self->vectors)
    {
        return -1;
    }
    for (size_t i = 0; i < self->capacity; i++)
    {
        self->vectors[i] = NULL;
    }

    self->metadata = (PyObject **)malloc(self->capacity * sizeof(PyObject *));
    if (!self->metadata)
    {
        free(self->vectors);
        return -1;
    }
    for (size_t i = 0; i < self->capacity; i++)
    {
        self->metadata[i] = NULL;
    }

    self->index = NULL;
    self->index_size = 0;

    return 0;
}

static void ffvec_dealloc(VectorSet *self)
{
    if (self->vectors)
    {
        for (size_t i = 0; i < self->size; i++)
        {
            free(self->vectors[i]);
            Py_XDECREF(self->metadata[i]);
        }
        free(self->vectors);
        free(self->metadata);
    }

    if (self->index)
    {
        for (size_t i = 0; i < self->index_size; i++)
        {
            metadata_index_dealloc(&self->index[i]);
        }
        free(self->index);
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *ffvec_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    VectorSet *self = (VectorSet *)type->tp_alloc(type, 0);
    if (!self)
    {
        return PyErr_NoMemory();
    }

    if (ffvec_init(self) != 0)
    {
        Py_DECREF(self);
        return NULL;
    }

    return (PyObject *)self;
}

static void ffvec_add(VectorSet *set, const unsigned char *vec, size_t len, PyObject *metadata)
{
    if (set->size == set->capacity)
    {
        size_t new_capacity = set->capacity + 10000;
        unsigned char **new_vectors = realloc(set->vectors, new_capacity * sizeof(unsigned char *));
        if (!new_vectors)
        {
            return;
        }
        set->vectors = new_vectors;

        PyObject **new_metadata = realloc(set->metadata, new_capacity * sizeof(PyObject *));
        if (!new_metadata)
        {
            return;
        }
        set->metadata = new_metadata;

        set->capacity = new_capacity;
    }

    set->vectors[set->size] = (unsigned char *)malloc(len);
    if (set->vectors[set->size] == NULL)
    {
        return;
    }
    memcpy(set->vectors[set->size], vec, len);

    Py_INCREF(metadata);
    set->metadata[set->size] = metadata;

    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(metadata, &pos, &key, &value))
    {
        int found = 0;
        for (size_t i = 0; i < set->index_size; i++)
        {
            if (PyObject_RichCompareBool(set->index[i].key, key, Py_EQ) &&
                PyObject_RichCompareBool(set->index[i].value, value, Py_EQ))
            {
                metadata_index_add(&set->index[i], key, value, set->size);
                found = 1;
                break;
            }
        }

        if (!found)
        {
            MetadataIndex *new_index = (MetadataIndex *)realloc(set->index, (set->index_size + 1) * sizeof(MetadataIndex));
            if (new_index == NULL)
            {
                return;
            }
            set->index = new_index;
            metadata_index_init(&set->index[set->index_size]);
            metadata_index_add(&set->index[set->index_size], key, value, set->size);
            set->index_size++;
        }
    }

    set->size++;
}

static PyObject *ffvec_add_py(PyObject *self, PyObject *args)
{
    VectorSet *set = (VectorSet *)self;
    PyObject *float_list;
    PyObject *metadata;
    if (!PyArg_ParseTuple(args, "OO", &float_list, &metadata))
    {
        return NULL;
    }

    if (!PyList_Check(float_list))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a list of floats");
        return NULL;
    }

    Py_ssize_t num_floats = PyList_Size(float_list);
    float *floats = (float *)malloc(num_floats * sizeof(float));
    if (!floats)
    {
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < num_floats; i++)
    {
        PyObject *item = PyList_GetItem(float_list, i);
        if (!PyFloat_Check(item))
        {
            free(floats);
            PyErr_SetString(PyExc_TypeError, "All items in the list must be floats");
            return NULL;
        }
        floats[i] = (float)PyFloat_AsDouble(item);
    }

    size_t quantized_len = num_floats / 8;
    unsigned char *quantized_vec = (unsigned char *)malloc(quantized_len);
    if (!quantized_vec)
    {
        free(floats);
        return PyErr_NoMemory();
    }

    quantize(floats, num_floats, quantized_vec);
    ffvec_add(set, quantized_vec, quantized_len, metadata);

    free(floats);
    free(quantized_vec);

    Py_RETURN_NONE;
}

static PyObject *ffvec_query_with_metadata_py(PyObject *self, PyObject *args)
{
    VectorSet *set = (VectorSet *)self;
    PyObject *query_obj, *metadata_filters;
    size_t top_k;
    if (!PyArg_ParseTuple(args, "OOn", &query_obj, &metadata_filters, &top_k))
    {
        return NULL;
    }

    if (!PySequence_Check(query_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Query must be a sequence of floats.");
        return NULL;
    }

    size_t len = VEC_SIZE;
    float *query_vec = (float *)malloc(len * sizeof(float));
    if (!query_vec)
    {
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < len; i++)
    {
        PyObject *item = PySequence_GetItem(query_obj, i);
        if (!PyFloat_Check(item))
        {
            free(query_vec);
            PyErr_SetString(PyExc_TypeError, "All items in the query must be floats.");
            return NULL;
        }
        query_vec[i] = (float)PyFloat_AsDouble(item);
    }

    unsigned char *quantized_query = (unsigned char *)malloc(len / 8);
    if (!quantized_query)
    {
        free(query_vec);
        return PyErr_NoMemory();
    }
    quantize(query_vec, len, quantized_query);

    size_t *filtered_indices = NULL;
    size_t filtered_size = 0;
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(metadata_filters, &pos, &key, &value))
    {
        for (size_t i = 0; i < set->index_size; i++)
        {
            if (PyObject_RichCompareBool(set->index[i].key, key, Py_EQ) &&
                PyObject_RichCompareBool(set->index[i].value, value, Py_EQ))
            {
                filtered_indices = (size_t *)realloc(filtered_indices, (filtered_size + set->index[i].size) * sizeof(size_t));
                if (!filtered_indices)
                {
                    free(query_vec);
                    free(quantized_query);
                    return PyErr_NoMemory();
                }
                memcpy(filtered_indices + filtered_size, set->index[i].indices, set->index[i].size * sizeof(size_t));
                filtered_size += set->index[i].size;
            }
        }
    }

    // Compute distances only for filtered vectors
    int *distances = (int *)malloc(filtered_size * sizeof(int));
    if (!distances)
    {
        free(query_vec);
        free(quantized_query);
        free(filtered_indices);
        return PyErr_NoMemory();
    }

#pragma omp parallel for
    for (size_t i = 0; i < filtered_size; i++)
    {
        size_t index = filtered_indices[i];
        distances[i] = hamming_distance(set->vectors[index], quantized_query, QUANTIZED_VEC_SIZE);
    }

    free(query_vec);
    free(quantized_query);

    PyObject *metadata_list = PyList_New(0);
    if (!metadata_list)
    {
        free(filtered_indices);
        free(distances);
        return PyErr_NoMemory();
    }

    for (size_t i = 0; i < top_k && i < filtered_size; i++)
    {
        size_t min_idx = 0;
        int min_dist = distances[0];
        for (size_t j = 1; j < filtered_size; j++)
        {
            if (distances[j] < min_dist)
            {
                min_idx = j;
                min_dist = distances[j];
            }
        }
        PyObject *metadata_item = set->metadata[filtered_indices[min_idx]];
        if (metadata_item == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Metadata item is NULL.");
            continue;
        }

        Py_INCREF(metadata_item); // Since we are appending it to a list
        PyList_Append(metadata_list, metadata_item);

        // Prepare for next iteration
        distances[min_idx] = INT_MAX;
    }

    free(filtered_indices);
    free(distances);

    return metadata_list;
}

static PyObject *ffvec_advanced_query_py(PyObject *self, PyObject *args)
{
    VectorSet *set = (VectorSet *)self;
    PyObject *metadata_filters, *query_obj;
    size_t top_k;
    if (!PyArg_ParseTuple(args, "OOn", &query_obj, &metadata_filters, &top_k))
    {
        return NULL;
    }

    if (!PySequence_Check(query_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Query must be a sequence of floats.");
        return NULL;
    }

    size_t len = VEC_SIZE; // Assume fixed length for simplicity
    float *query_vec = (float *)malloc(len * sizeof(float));
    if (!query_vec)
    {
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < len; i++)
    {
        PyObject *item = PySequence_GetItem(query_obj, i);
        if (!PyFloat_Check(item))
        {
            free(query_vec);
            PyErr_SetString(PyExc_TypeError, "All items in the query must be floats.");
            return NULL;
        }
        query_vec[i] = (float)PyFloat_AsDouble(item);
    }

    // Initialize weights for SGD
    float *weights = (float *)calloc(len, sizeof(float));
    if (!weights)
    {
        free(query_vec);
        return PyErr_NoMemory();
    }

    float learning_rate = 0.01;
    int iterations = 10;
    for (int iter = 0; iter < iterations; iter++)
    {
        sgd_update(weights, query_vec, 1.0, len, learning_rate * 99);
        for (int n = 0; n < 99; n++)
        {
            size_t idx = rand() % set->size;
            float *vec = (float *)malloc(len * sizeof(float));
            dequantize(set->vectors[idx], VEC_SIZE, vec);
            sgd_update(weights, vec, 0.0, len, learning_rate);
        }
    }

    // Quantize the resulting weights
    unsigned char *quantized_weights = (unsigned char *)malloc(len / 8);
    if (!quantized_weights)
    {
        free(weights);
        free(query_vec);
        return PyErr_NoMemory();
    }
    quantize(weights, len, quantized_weights);

    // Compute Hamming distances for filtered vectors
    size_t *filtered_indices = NULL;
    size_t filtered_size = 0;
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    while (PyDict_Next(metadata_filters, &pos, &key, &value))
    {
        for (size_t i = 0; i < set->index_size; i++)
        {
            if (PyObject_RichCompareBool(set->index[i].key, key, Py_EQ) &&
                PyObject_RichCompareBool(set->index[i].value, value, Py_EQ))
            {
                filtered_indices = (size_t *)realloc(filtered_indices, (filtered_size + set->index[i].size) * sizeof(size_t));
                if (!filtered_indices)
                {
                    free(query_vec);
                    free(weights);
                    free(quantized_weights);
                    return PyErr_NoMemory();
                }
                memcpy(filtered_indices + filtered_size, set->index[i].indices, set->index[i].size * sizeof(size_t));
                filtered_size += set->index[i].size;
            }
        }
    }

    int *distances = (int *)malloc(filtered_size * sizeof(int));
    if (!distances)
    {
        free(query_vec);
        free(quantized_weights);
        free(weights);
        free(filtered_indices);
        return PyErr_NoMemory();
    }

#pragma omp parallel for
    for (size_t i = 0; i < filtered_size; i++)
    {
        size_t index = filtered_indices[i];
        distances[i] = hamming_distance(set->vectors[index], quantized_weights, QUANTIZED_VEC_SIZE);
    }

    // Sort and return the results based on distances
    PyObject *metadata_list = PyList_New(0);
    if (!metadata_list)
    {
        free(query_vec);
        free(filtered_indices);
        free(distances);
        return PyErr_NoMemory();
    }

    for (size_t i = 0; i < top_k && i < filtered_size; i++)
    {
        size_t min_idx = 0;
        int min_dist = distances[0];
        for (size_t j = 1; j < filtered_size; j++)
        {
            if (distances[j] < min_dist)
            {
                min_idx = j;
                min_dist = distances[j];
            }
        }
        PyObject *metadata_item = set->metadata[filtered_indices[min_idx]];
        if (metadata_item == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Metadata item is NULL.");
            continue;
        }

        Py_INCREF(metadata_item); // Since we are appending it to a list
        PyList_Append(metadata_list, metadata_item);

        distances[min_idx] = INT_MAX;
    }

    free(filtered_indices);
    free(distances);
    free(query_vec);

    return metadata_list;
}

static PyMethodDef VectorSet_methods[] = {
    {"add", (PyCFunction)ffvec_add_py, METH_VARARGS, "Add a vector to the set"},
    {"query_with_metadata", (PyCFunction)ffvec_query_with_metadata_py, METH_VARARGS, "Query the vector set based on Hamming distance and metadata filters"},
    {"advanced_query", (PyCFunction)ffvec_advanced_query_py, METH_VARARGS, "Query the vector set based on Hamming distance and metadata filters"},
    {NULL}};

static PyTypeObject VectorSetType = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "ffvec.VectorSet",
    .tp_doc = "Vector sets",
    .tp_basicsize = sizeof(VectorSet),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = ffvec_new,
    .tp_dealloc = (destructor)ffvec_dealloc,
    .tp_methods = VectorSet_methods};

static PyModuleDef vectorsetmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "ffvec",
    .m_doc = "Module for a vector DB supporting binary vectors and a logistic regression based query mechanism",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_ffvec(void)
{
    if (PyType_Ready(&VectorSetType) < 0)
        return NULL;

    PyObject *m = PyModule_Create(&vectorsetmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&VectorSetType);
    PyModule_AddObject(m, "VectorSet", (PyObject *)&VectorSetType);

    return m;
}