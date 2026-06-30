import qnsim as qs


def test_top_level_frontend_surface():
    program = qs.Program(2, 2)
    program.add(qs.ops.H, 0)
    program.add(qs.ops.CZ, (0, 1))
    program.add(qs.ops.RX(0.1), 0)
    program.add_measurement(0, 0)
    program.add_measurement(1, 1)

    assert len(program.operations) == 5
    assert program.operations[0].operation.name == "H"
    assert isinstance(program.operations[3], qs.Measurement)


def test_register_types_exposed():
    qr = qs.QuantumRegister(2, name="q")
    assert isinstance(qr[0], qs.RegisterRef)
