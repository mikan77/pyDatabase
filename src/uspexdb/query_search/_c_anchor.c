#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    Py_buffer view;
    const char *name;
    Py_ssize_t itemsize;
} ReadBuffer;


static int get_ro_buffer(PyObject *obj, ReadBuffer *buf, Py_ssize_t itemsize, const char *name) {
    buf->name = name;
    buf->itemsize = itemsize;
    if (PyObject_GetBuffer(obj, &buf->view, PyBUF_CONTIG_RO) < 0) {
        return -1;
    }
    if (buf->view.len % itemsize != 0) {
        PyErr_Format(PyExc_ValueError, "%s byte length is not divisible by item size", name);
        PyBuffer_Release(&buf->view);
        return -1;
    }
    return 0;
}


static Py_ssize_t buf_len(const ReadBuffer *buf) {
    return buf->view.len / buf->itemsize;
}


static void release_buffers(ReadBuffer *buffers, int count) {
    for (int i = 0; i < count; i++) {
        if (buffers[i].view.buf != NULL) {
            PyBuffer_Release(&buffers[i].view);
            buffers[i].view.buf = NULL;
        }
    }
}


static int contains_u32(const uint32_t *values, int64_t start, int64_t stop, uint32_t needle) {
    for (int64_t i = start; i < stop; i++) {
        if (values[i] == needle) {
            return 1;
        }
    }
    return 0;
}


static PyObject *py_prefilter_candidates(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *status_obj;
    PyObject *elem_offsets_obj;
    PyObject *elem_numbers_obj;
    PyObject *elem_counts_obj;
    PyObject *edge_offsets_obj;
    PyObject *edge_codes_obj;
    PyObject *req_numbers_obj;
    PyObject *req_counts_obj;
    PyObject *req_edges_obj;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOO",
            &status_obj,
            &elem_offsets_obj,
            &elem_numbers_obj,
            &elem_counts_obj,
            &edge_offsets_obj,
            &edge_codes_obj,
            &req_numbers_obj,
            &req_counts_obj,
            &req_edges_obj)) {
        return NULL;
    }

    ReadBuffer buffers[9] = {0};
    if (get_ro_buffer(status_obj, &buffers[0], 1, "status") < 0 ||
        get_ro_buffer(elem_offsets_obj, &buffers[1], 8, "element_offsets") < 0 ||
        get_ro_buffer(elem_numbers_obj, &buffers[2], 1, "element_numbers") < 0 ||
        get_ro_buffer(elem_counts_obj, &buffers[3], 2, "element_counts") < 0 ||
        get_ro_buffer(edge_offsets_obj, &buffers[4], 8, "edge_offsets") < 0 ||
        get_ro_buffer(edge_codes_obj, &buffers[5], 4, "edge_codes") < 0 ||
        get_ro_buffer(req_numbers_obj, &buffers[6], 1, "required_numbers") < 0 ||
        get_ro_buffer(req_counts_obj, &buffers[7], 2, "required_counts") < 0 ||
        get_ro_buffer(req_edges_obj, &buffers[8], 4, "required_edges") < 0) {
        release_buffers(buffers, 9);
        return NULL;
    }

    Py_ssize_t n_structures = buf_len(&buffers[0]);
    if (buf_len(&buffers[1]) != n_structures + 1 || buf_len(&buffers[4]) != n_structures + 1) {
        release_buffers(buffers, 9);
        PyErr_SetString(PyExc_ValueError, "offset arrays must have length len(status) + 1");
        return NULL;
    }
    if (buf_len(&buffers[6]) != buf_len(&buffers[7])) {
        release_buffers(buffers, 9);
        PyErr_SetString(PyExc_ValueError, "required_numbers and required_counts length mismatch");
        return NULL;
    }

    const uint8_t *status = (const uint8_t *)buffers[0].view.buf;
    const int64_t *elem_offsets = (const int64_t *)buffers[1].view.buf;
    const uint8_t *elem_numbers = (const uint8_t *)buffers[2].view.buf;
    const uint16_t *elem_counts = (const uint16_t *)buffers[3].view.buf;
    const int64_t *edge_offsets = (const int64_t *)buffers[4].view.buf;
    const uint32_t *edge_codes = (const uint32_t *)buffers[5].view.buf;
    const uint8_t *req_numbers = (const uint8_t *)buffers[6].view.buf;
    const uint16_t *req_counts = (const uint16_t *)buffers[7].view.buf;
    const uint32_t *req_edges = (const uint32_t *)buffers[8].view.buf;
    Py_ssize_t n_req_numbers = buf_len(&buffers[6]);
    Py_ssize_t n_req_edges = buf_len(&buffers[8]);

    PyObject *result = PyList_New(0);
    if (result == NULL) {
        release_buffers(buffers, 9);
        return NULL;
    }

    for (Py_ssize_t structure_idx = 0; structure_idx < n_structures; structure_idx++) {
        if (status[structure_idx] != 1) {
            continue;
        }

        int ok = 1;
        int64_t elem_start = elem_offsets[structure_idx];
        int64_t elem_stop = elem_offsets[structure_idx + 1];
        for (Py_ssize_t req_idx = 0; req_idx < n_req_numbers && ok; req_idx++) {
            uint8_t number = req_numbers[req_idx];
            uint16_t required = req_counts[req_idx];
            uint16_t have = 0;
            for (int64_t elem_idx = elem_start; elem_idx < elem_stop; elem_idx++) {
                if (elem_numbers[elem_idx] == number) {
                    have = elem_counts[elem_idx];
                    break;
                }
            }
            if (have < required) {
                ok = 0;
            }
        }
        if (!ok) {
            continue;
        }

        int64_t edge_start = edge_offsets[structure_idx];
        int64_t edge_stop = edge_offsets[structure_idx + 1];
        for (Py_ssize_t req_idx = 0; req_idx < n_req_edges && ok; req_idx++) {
            if (!contains_u32(edge_codes, edge_start, edge_stop, req_edges[req_idx])) {
                ok = 0;
            }
        }
        if (!ok) {
            continue;
        }

        PyObject *value = PyLong_FromSsize_t(structure_idx);
        if (value == NULL) {
            Py_DECREF(result);
            release_buffers(buffers, 9);
            return NULL;
        }
        if (PyList_Append(result, value) < 0) {
            Py_DECREF(value);
            Py_DECREF(result);
            release_buffers(buffers, 9);
            return NULL;
        }
        Py_DECREF(value);
    }

    release_buffers(buffers, 9);
    return result;
}


typedef struct {
    int qn;
    int tn;
    const uint8_t *q_atomic;
    const uint8_t *q_hyb;
    const int64_t *q_offsets;
    const int32_t *q_neighbors;
    const uint8_t *q_orders;
    const int32_t *q_order;
    const uint8_t *t_atomic;
    const uint8_t *t_hyb;
    const int64_t *t_offsets;
    const int32_t *t_neighbors;
    const uint8_t *t_orders;
    int strict_bonds;
    int strict_atom_types;
    int allow_hydrogen_wildcards;
    int32_t *mapping;
    uint8_t *used;
    uint8_t *candidate_ok;
    PyObject *results;
} MatchState;


static int hyb_matches(uint8_t target, uint8_t query) {
    if (query == 0) {
        return 1;
    }
    if (target == 0) {
        return 0;
    }
    if (query == 4) {
        return target == 4 || target == 2;
    }
    if (query == 2) {
        return target == 2 || target == 4;
    }
    return target == query;
}


static int node_matches(MatchState *state, int q, int t) {
    uint8_t q_atomic = state->q_atomic[q];
    if (q_atomic == 0) {
        return 1;
    }
    if (state->allow_hydrogen_wildcards && q_atomic == 1) {
        return 1;
    }
    if (state->t_atomic[t] != q_atomic) {
        return 0;
    }
    if (state->strict_atom_types && !hyb_matches(state->t_hyb[t], state->q_hyb[q])) {
        return 0;
    }
    return 1;
}


static int target_edge_order(MatchState *state, int a, int b, uint8_t *order_out) {
    int64_t start = state->t_offsets[a];
    int64_t stop = state->t_offsets[a + 1];
    for (int64_t i = start; i < stop; i++) {
        if (state->t_neighbors[i] == b) {
            *order_out = state->t_orders[i];
            return 1;
        }
    }
    return 0;
}


static int query_edge_order(MatchState *state, int a, int b, uint8_t *order_out) {
    int64_t start = state->q_offsets[a];
    int64_t stop = state->q_offsets[a + 1];
    for (int64_t i = start; i < stop; i++) {
        if (state->q_neighbors[i] == b) {
            *order_out = state->q_orders[i];
            return 1;
        }
    }
    return 0;
}


static int edge_matches(uint8_t target_order, uint8_t query_order, int strict_bonds) {
    if (!strict_bonds) {
        return 1;
    }
    if (query_order == 0 || query_order == 7 || query_order == 8) {
        return 1;
    }
    if (target_order == query_order) {
        return 1;
    }
    if (query_order == 4 && (target_order == 4 || target_order == 0)) {
        return 1;
    }
    return target_order == 0;
}


static int append_mapping_tuple(MatchState *state) {
    PyObject *tuple = PyTuple_New(state->qn);
    if (tuple == NULL) {
        return -1;
    }
    for (int i = 0; i < state->qn; i++) {
        PyObject *value = PyLong_FromLong((long)state->mapping[i]);
        if (value == NULL) {
            Py_DECREF(tuple);
            return -1;
        }
        PyTuple_SET_ITEM(tuple, i, value);
    }
    if (PyList_Append(state->results, tuple) < 0) {
        Py_DECREF(tuple);
        return -1;
    }
    Py_DECREF(tuple);
    return 0;
}


static int recurse_match(MatchState *state, int position) {
    if (position >= state->qn) {
        return append_mapping_tuple(state);
    }

    int q = state->q_order[position];
    for (int t = 0; t < state->tn; t++) {
        if (state->used[t]) {
            continue;
        }
        if (!state->candidate_ok[(Py_ssize_t)q * state->tn + t]) {
            continue;
        }

        int ok = 1;
        int64_t q_start = state->q_offsets[q];
        int64_t q_stop = state->q_offsets[q + 1];
        for (int64_t qi = q_start; qi < q_stop; qi++) {
            int q_neighbor = state->q_neighbors[qi];
            int mapped_target = state->mapping[q_neighbor];
            if (mapped_target < 0) {
                continue;
            }
            uint8_t t_order = 0;
            uint8_t q_order = 0;
            if (!target_edge_order(state, t, mapped_target, &t_order)) {
                ok = 0;
                break;
            }
            if (!query_edge_order(state, q, q_neighbor, &q_order)) {
                ok = 0;
                break;
            }
            if (!edge_matches(t_order, q_order, state->strict_bonds)) {
                ok = 0;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        state->mapping[q] = t;
        state->used[t] = 1;
        if (recurse_match(state, position + 1) < 0) {
            return -1;
        }
        state->used[t] = 0;
        state->mapping[q] = -1;
    }
    return 0;
}


static PyObject *py_match_fragment(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *q_atomic_obj;
    PyObject *q_hyb_obj;
    PyObject *q_offsets_obj;
    PyObject *q_neighbors_obj;
    PyObject *q_orders_obj;
    PyObject *q_order_obj;
    PyObject *t_atomic_obj;
    PyObject *t_hyb_obj;
    PyObject *t_offsets_obj;
    PyObject *t_neighbors_obj;
    PyObject *t_orders_obj;
    int strict_bonds;
    int strict_atom_types;
    int allow_hydrogen_wildcards;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOiii",
            &q_atomic_obj,
            &q_hyb_obj,
            &q_offsets_obj,
            &q_neighbors_obj,
            &q_orders_obj,
            &q_order_obj,
            &t_atomic_obj,
            &t_hyb_obj,
            &t_offsets_obj,
            &t_neighbors_obj,
            &t_orders_obj,
            &strict_bonds,
            &strict_atom_types,
            &allow_hydrogen_wildcards)) {
        return NULL;
    }

    ReadBuffer buffers[11] = {0};
    if (get_ro_buffer(q_atomic_obj, &buffers[0], 1, "query_atomic_numbers") < 0 ||
        get_ro_buffer(q_hyb_obj, &buffers[1], 1, "query_hybridization") < 0 ||
        get_ro_buffer(q_offsets_obj, &buffers[2], 8, "query_adj_offsets") < 0 ||
        get_ro_buffer(q_neighbors_obj, &buffers[3], 4, "query_neighbors") < 0 ||
        get_ro_buffer(q_orders_obj, &buffers[4], 1, "query_edge_orders") < 0 ||
        get_ro_buffer(q_order_obj, &buffers[5], 4, "query_match_order") < 0 ||
        get_ro_buffer(t_atomic_obj, &buffers[6], 1, "target_atomic_numbers") < 0 ||
        get_ro_buffer(t_hyb_obj, &buffers[7], 1, "target_hybridization") < 0 ||
        get_ro_buffer(t_offsets_obj, &buffers[8], 8, "target_adj_offsets") < 0 ||
        get_ro_buffer(t_neighbors_obj, &buffers[9], 4, "target_neighbors") < 0 ||
        get_ro_buffer(t_orders_obj, &buffers[10], 1, "target_edge_orders") < 0) {
        release_buffers(buffers, 11);
        return NULL;
    }

    Py_ssize_t qn = buf_len(&buffers[0]);
    Py_ssize_t tn = buf_len(&buffers[6]);
    if (qn <= 0 || tn < 0 || qn > 128) {
        release_buffers(buffers, 11);
        PyErr_SetString(PyExc_ValueError, "unsupported query/target size");
        return NULL;
    }
    if (buf_len(&buffers[1]) != qn || buf_len(&buffers[2]) != qn + 1 ||
        buf_len(&buffers[5]) != qn || buf_len(&buffers[7]) != tn ||
        buf_len(&buffers[8]) != tn + 1) {
        release_buffers(buffers, 11);
        PyErr_SetString(PyExc_ValueError, "array length mismatch in match_fragment inputs");
        return NULL;
    }

    MatchState state;
    memset(&state, 0, sizeof(state));
    state.qn = (int)qn;
    state.tn = (int)tn;
    state.q_atomic = (const uint8_t *)buffers[0].view.buf;
    state.q_hyb = (const uint8_t *)buffers[1].view.buf;
    state.q_offsets = (const int64_t *)buffers[2].view.buf;
    state.q_neighbors = (const int32_t *)buffers[3].view.buf;
    state.q_orders = (const uint8_t *)buffers[4].view.buf;
    state.q_order = (const int32_t *)buffers[5].view.buf;
    state.t_atomic = (const uint8_t *)buffers[6].view.buf;
    state.t_hyb = (const uint8_t *)buffers[7].view.buf;
    state.t_offsets = (const int64_t *)buffers[8].view.buf;
    state.t_neighbors = (const int32_t *)buffers[9].view.buf;
    state.t_orders = (const uint8_t *)buffers[10].view.buf;
    state.strict_bonds = strict_bonds;
    state.strict_atom_types = strict_atom_types;
    state.allow_hydrogen_wildcards = allow_hydrogen_wildcards;

    state.mapping = (int32_t *)malloc((size_t)state.qn * sizeof(int32_t));
    state.used = (uint8_t *)calloc((size_t)(state.tn > 0 ? state.tn : 1), sizeof(uint8_t));
    state.candidate_ok = (uint8_t *)calloc((size_t)state.qn * (size_t)(state.tn > 0 ? state.tn : 1), sizeof(uint8_t));
    if (state.mapping == NULL || state.used == NULL || state.candidate_ok == NULL) {
        free(state.mapping);
        free(state.used);
        free(state.candidate_ok);
        release_buffers(buffers, 11);
        return PyErr_NoMemory();
    }
    for (int i = 0; i < state.qn; i++) {
        state.mapping[i] = -1;
    }
    for (int q = 0; q < state.qn; q++) {
        for (int t = 0; t < state.tn; t++) {
            state.candidate_ok[(Py_ssize_t)q * state.tn + t] = (uint8_t)node_matches(&state, q, t);
        }
    }

    state.results = PyList_New(0);
    if (state.results == NULL) {
        free(state.mapping);
        free(state.used);
        free(state.candidate_ok);
        release_buffers(buffers, 11);
        return NULL;
    }

    if (recurse_match(&state, 0) < 0) {
        Py_DECREF(state.results);
        free(state.mapping);
        free(state.used);
        free(state.candidate_ok);
        release_buffers(buffers, 11);
        return NULL;
    }

    PyObject *result = state.results;
    free(state.mapping);
    free(state.used);
    free(state.candidate_ok);
    release_buffers(buffers, 11);
    return result;
}


static PyMethodDef Methods[] = {
    {
        "prefilter_candidates",
        py_prefilter_candidates,
        METH_VARARGS,
        "Return compact-cache row indexes matching sparse element and edge requirements.",
    },
    {
        "match_fragment",
        py_match_fragment,
        METH_VARARGS,
        "Return target atom tuples matching a query graph against one target graph.",
    },
    {NULL, NULL, 0, NULL},
};


static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_c_anchor",
    "CPython/C anchor matcher for compact USPEX contact graphs.",
    -1,
    Methods,
};


PyMODINIT_FUNC PyInit__c_anchor(void) {
    return PyModule_Create(&Module);
}
