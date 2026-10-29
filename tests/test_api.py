"""End-to-end API flow: admin -> hospital -> donor -> recommendation -> notify -> respond -> donation."""
import uuid

import pytest

GN = {"lat": 28.4744, "lon": 77.5040}


def _hdr(token):
    return {"Authorization": f"Bearer {token}"}


def _login(client, email, password):
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture(scope="module")
def admin(client):
    return _login(client, "admin@example.com", "AdminPass123")


@pytest.fixture(scope="module")
def hospital(client, admin):
    email = f"hosp-{uuid.uuid4().hex[:6]}@example.com"
    r = client.post("/auth/users", headers=_hdr(admin),
                    json={"email": email, "password": "HospitalPass1", "name": "Test Hospital", "role": "hospital"})
    assert r.status_code == 201, r.text
    return _login(client, email, "HospitalPass1")


def test_health_and_dashboard(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200


def test_auth_rules(client, admin):
    assert client.get("/auth/me").status_code == 401
    assert client.post("/auth/login", json={"email": "nobody@example.com", "password": "x" * 8}).status_code == 401
    assert client.get("/auth/me", headers=_hdr("garbage")).status_code == 401
    # public registration only creates donors, and cannot create staff
    email = f"d-{uuid.uuid4().hex[:6]}@example.com"
    r = client.post("/auth/register", json={"email": email, "password": "DonorPass12", "name": "Reg Donor"})
    assert r.status_code == 201 and r.json()["role"] == "donor"
    assert client.post("/auth/register", json={"email": email, "password": "DonorPass12", "name": "x"}).status_code == 409
    assert client.post("/auth/users", headers=_hdr(r.json()["access_token"]),
                       json={"email": "x@example.com", "password": "Password123", "name": "x", "role": "admin"}).status_code == 403
    assert client.post("/auth/register", json={"email": "bad", "password": "short", "name": "x"}).status_code == 422


def test_role_separation(client, hospital):
    assert client.post("/admin/seed-demo", headers=_hdr(hospital)).status_code == 403
    assert client.get("/donors", headers=_hdr(hospital)).status_code == 403
    assert client.get("/donors/me", headers=_hdr(hospital)).status_code == 403


def test_full_emergency_flow(client, admin, hospital):
    # 1. demo (synthetic) donors
    r = client.post("/admin/seed-demo?n_donors=300", headers=_hdr(admin))
    assert r.status_code == 200 and r.json()["inserted"] == 300

    # 2. a real donor registers and creates a consenting profile next to the hospital
    email = f"flow-{uuid.uuid4().hex[:6]}@example.com"
    dtoken = client.post("/auth/register", json={"email": email, "password": "DonorPass12",
                                                 "name": "Flow Donor"}).json()["access_token"]
    prof = {"display_name": "Flow Donor Kumar", "blood_group": "O+", "sex": "M", "age": 29, "weight_kg": 72,
            "city": "Greater Noida", "phone": "+910000000000", "consent_to_contact": True, **GN}
    r = client.post("/donors/me", headers=_hdr(dtoken), json=prof)
    assert r.status_code == 201, r.text
    donor_id = r.json()["id"]
    assert client.post("/donors/me", headers=_hdr(dtoken), json=prof).status_code == 409

    # 3. hospital raises a critical request
    r = client.post("/requests", headers=_hdr(hospital), json={
        "hospital_name": "Demo Hospital", "patient_blood_group": "O+", "units_needed": 1,
        "urgency": "critical", "city": "Greater Noida", "max_radius_km": 30, **GN})
    assert r.status_code == 201, r.text
    req_id = r.json()["id"]

    # 4. ranked recommendations with explanations; phone / exact location never exposed
    r = client.get(f"/requests/{req_id}/recommendations?top_k=50&mode=weighted&include_excluded=true", headers=_hdr(hospital))
    assert r.status_code == 200, r.text
    body = r.json()
    recs = body["recommendations"]
    assert len(recs) > 5 and body["stats"]["eligible"] >= len(recs)
    scores = [x["score"] for x in recs]
    assert scores == sorted(scores, reverse=True) and [x["rank"] for x in recs] == list(range(1, len(recs) + 1))
    assert all(x["explanation"] for x in recs)
    ours = next(x for x in recs if x["donor_id"] == donor_id)
    assert ours["name"] == "Flow K." and ours["distance_km"] < 1
    text = r.text
    assert "phone" not in text and "+9100" not in text and '"lat"' not in text
    assert abs(sum(body["weights"].values()) - 1) < 0.01

    # 5. cannot contact an arbitrary donor outside the candidate list
    assert client.post(f"/requests/{req_id}/notify", headers=_hdr(hospital), json={"donor_ids": [999999]}).status_code == 400

    # 6. notify our donor, who then accepts
    r = client.post(f"/requests/{req_id}/notify", headers=_hdr(hospital), json={"donor_ids": [donor_id]})
    assert r.status_code == 201, r.text
    note_id = r.json()[0]["id"]
    # already-contacted donors disappear from the candidate list
    again = client.get(f"/requests/{req_id}/recommendations?top_k=50&mode=weighted", headers=_hdr(hospital)).json()
    assert donor_id not in [x["donor_id"] for x in again["recommendations"]]

    mine = client.get("/donors/me/notifications", headers=_hdr(dtoken)).json()
    assert mine[0]["id"] == note_id and mine[0]["accepted"] is None
    assert client.post(f"/notifications/{note_id}/respond", headers=_hdr(hospital), json={"accepted": True}).status_code == 403
    r = client.post(f"/notifications/{note_id}/respond", headers=_hdr(dtoken), json={"accepted": True})
    assert r.status_code == 200 and r.json()["accepted"] is True
    assert client.post(f"/notifications/{note_id}/respond", headers=_hdr(dtoken), json={"accepted": False}).status_code == 409

    # 7. hospital confirms the donation -> reliability stats + request status updated
    r = client.post(f"/notifications/{note_id}/donation", headers=_hdr(hospital), json={"completed": True})
    assert r.status_code == 200 and r.json()["donated"] is True
    me = client.get("/donors/me", headers=_hdr(dtoken)).json()
    assert (me["total_contacted"], me["total_responded"], me["total_accepted"], me["total_donations"]) == (1, 1, 1, 1)
    assert me["last_donation_date"] is not None
    assert client.get(f"/requests/{req_id}", headers=_hdr(hospital)).json()["status"] == "fulfilled"
    assert client.post(f"/notifications/{note_id}/donation", headers=_hdr(hospital), json={"completed": True}).status_code == 409

    # 8. a donor who just donated is no longer eligible for a new request
    r2 = client.post("/requests", headers=_hdr(hospital), json={
        "hospital_name": "Demo Hospital", "patient_blood_group": "O+", "urgency": "urgent", **GN}).json()["id"]
    ex = client.get(f"/requests/{r2}/recommendations?top_k=50&include_excluded=true", headers=_hdr(hospital)).json()
    assert donor_id not in [x["donor_id"] for x in ex["recommendations"]]
    assert any(e["donor_id"] == donor_id and "eligible again" in " ".join(e["reasons"]) for e in ex["excluded"])

    # 9. availability update + erasure
    assert client.put("/donors/me/availability", headers=_hdr(dtoken), json={"status": "busy"}).json()["availability_status"] == "busy"
    assert client.delete("/donors/me", headers=_hdr(dtoken)).status_code == 204
    assert client.get("/donors/me", headers=_hdr(dtoken)).status_code == 404


def test_hospitals_cannot_see_each_others_requests(client, admin, hospital):
    other_email = f"hosp2-{uuid.uuid4().hex[:6]}@example.com"
    client.post("/auth/users", headers=_hdr(admin),
                json={"email": other_email, "password": "HospitalPass2", "name": "Other", "role": "hospital"})
    other = _login(client, other_email, "HospitalPass2")
    rid = client.post("/requests", headers=_hdr(hospital), json={
        "hospital_name": "H1", "patient_blood_group": "A+", **GN}).json()["id"]
    assert client.get(f"/requests/{rid}", headers=_hdr(other)).status_code == 404
    assert client.get(f"/requests/{rid}/recommendations", headers=_hdr(other)).status_code == 404
    assert client.get(f"/requests/{rid}", headers=_hdr(admin)).status_code == 200


def test_validation_errors(client, hospital):
    bad = {"hospital_name": "H", "patient_blood_group": "Z+", **GN}
    assert client.post("/requests", headers=_hdr(hospital), json=bad).status_code == 422
    bad = {"hospital_name": "H", "patient_blood_group": "A+", "lat": 123, "lon": 0}
    assert client.post("/requests", headers=_hdr(hospital), json=bad).status_code == 422
    assert client.get("/requests/999999", headers=_hdr(hospital)).status_code == 404


def test_admin_endpoints(client, admin):
    s = client.get("/admin/stats", headers=_hdr(admin)).json()
    assert s["donors"] >= 300 and s["synthetic_donors"] >= 300
    assert client.get("/admin/model", headers=_hdr(admin)).status_code == 200
    page = client.get("/donors?limit=5", headers=_hdr(admin)).json()
    assert len(page) == 5 and "phone" not in page[0] and page[0]["is_synthetic"] is True
