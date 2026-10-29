# Major Project Synopsis (Phase-I)

## Trust-Aware Recommendation of Emergency Blood Donors Using Reliability and Availability Indicators

**Degree:** Bachelor of Technology in Computer Science and Engineering - Session 2026-27  
**School:** School of Computer Science and Engineering, IILM University, Greater Noida  
**Guide:** Mrs. Madhavi Tripathi  
**Submitted:** May 2026

| Team member | Roll no. |
|---|---|
| Aniket Kumar | 2410031618 |
| Aditya Pandey | 2410031650 |
| Shivam Raj | 2410031707 |
| Divyanshu Kumar | 2410031735 |

---

## Abstract

During medical emergencies, finding a suitable blood donor quickly is critical. A blood-group match alone does not
guarantee that a registered donor is currently available, reachable, eligible, or likely to respond. This project
proposes a Trust-Aware Emergency Blood Donor Recommendation System that ranks compatible donors using reliability,
current availability, eligibility, geographic proximity, and request urgency.

The system first filters donors using basic compatibility and eligibility rules and then applies a composite
trust-aware score to prioritise donors who are more likely to respond successfully. The proposed prototype can use
Python, Flask/FastAPI, a database, and Scikit-learn, with weighted scoring as the initial approach and
machine-learning models such as Random Forest or XGBoost when sufficient training data are available. The expected
outcome is a ranked and interpretable donor list that can reduce donor-search time and improve emergency blood
coordination.

## 1. Problem Definition

During medical emergencies, identifying a suitable blood donor within a limited time is a critical challenge.
Traditional donor-search methods often depend on static donor databases, manual calls, personal contacts, or public
appeals, which may not reflect whether a registered donor is currently reachable or willing to respond. A donor may
be compatible by blood group but unavailable because of recent donation, temporary ineligibility, distance,
inactivity, or lack of response. Therefore, a simple blood-group matching mechanism is insufficient for emergency
donor recommendation.

The proposed project aims to develop a trust-aware recommendation system that ranks compatible donors using
measurable reliability and availability indicators along with proximity and blood-group compatibility. The system
will prioritise donors who are more likely to be reachable, eligible, and responsive, thereby reducing donor-search
time and improving emergency coordination.

## 2. Literature Summary

Recent research demonstrates growing use of machine learning, algorithmic matching, intelligent donor management, and
real-time systems for improving blood donation. The following studies are particularly relevant to the proposed
trust-aware recommendation approach.

| Year | Research paper | Key contribution |
|---|---|---|
| 2022 | Predicting willingness to donate blood based on machine learning: two blood donor recruitments during COVID-19 outbreaks [1] | Used seven ML models and 95,476 recruitment records to rank donor willingness. Donation interval, age and donation frequency were among the strongest predictors; XGBoost and SVM performed strongly. |
| 2022 | Machine learning-based prediction of fainting during blood donations using donor properties and weather data as features [2] | Demonstrated that donor attributes and contextual data can be used with ML to predict donation-related outcomes, supporting data-driven donor management. |
| 2023 | Matching algorithms for blood donation [3] | Developed automated donor-opportunity matching policies. Simulations and a real pilot showed that algorithmic matching can increase donor action and improve coordination. |
| 2023 | A Location-Based Blood Donor Locator System Using Intelligent Matching Methodology [4] | Proposed digital donor registration, blood-group matching, geographic donor identification and emergency request management to reduce donor-search time. |
| 2024 | A Machine Learning-Based Blood Donor Recommendation System to Enhance Blood Donation Efficiency [5] | Applied Random Forest for blood-donor recommendation using factors such as blood group, location and donor availability. |
| 2025 | Real-Time Blood Donor Prediction and Campaign Optimization using DNN-Autoencoder and MERN Architecture [6] | Combined a DNN-Autoencoder with a real-time web architecture for donor prediction, blood-group matching, scheduling and donor availability optimisation. |
| 2025/2026 | Artificial intelligence in blood donor management: A narrative review [7] | Reviews AI applications for donor recruitment, retention and management, while highlighting data limitations, ethics, cybersecurity and implementation challenges. |

## 3. Research Gap Identified from Literature

The literature shows that blood-donor systems have progressed from simple registration and location-based search
toward machine-learning prediction and algorithmic matching. However, most approaches emphasise one or two
dimensions, such as willingness to donate, geographic proximity, blood-group compatibility, or general availability.
A practical emergency system needs to combine these dimensions into a single ranking mechanism. In particular, the
distinction between a donor who is merely registered and a donor who is consistently reliable and currently available
is not sufficiently addressed in many existing systems.

There is therefore an opportunity to introduce a trust-aware score that uses historical response/donation behaviour,
current availability, eligibility, proximity and request urgency. The proposed project addresses this gap by treating
reliability and availability as first-class recommendation indicators rather than secondary filters.

## 4. Objectives

1. To design and implement a trust-aware emergency blood-donor recommendation framework that combines blood-group
   compatibility, availability, reliability and geographic proximity.
2. To develop a donor-ranking score using historical donation/response behaviour and current availability indicators
   to prioritise donors who are more likely to respond successfully.
3. To reduce the time and effort required to identify suitable emergency donors and evaluate the recommendation system
   using ranking quality, response prediction and retrieval-time measures.

## 5. Scope

The proposed work covers donor registration, donor-profile management, emergency blood requests, compatibility
filtering, availability estimation, reliability scoring and ranked donor recommendation. Donor records may contain
blood group, approximate location, donation history, response history, recent donation date, current availability and
other non-sensitive indicators required for the prototype. The recommendation engine first removes donors who fail
basic compatibility or eligibility rules and then ranks remaining candidates using a composite trust score. The system
may provide separate interfaces for authorised hospitals/blood banks and donors, including emergency notifications and
request status.

Real-time integration with national blood-bank systems, clinical eligibility certification, blood testing, medical
diagnosis and actual transfusion decisions are **outside the scope** of this academic prototype.

## 6. Novelty

The main novelty is the introduction of a trust-aware donor ranking layer for emergency recommendations. Instead of
returning all compatible donors equally or sorting only by distance, the system calculates a composite score from
reliability, availability, eligibility, compatibility, proximity and urgency. Reliability can reflect historical
response rate, successful donation history, recent activity and consistency of participation, while availability
represents the donor's current or predicted ability to respond. This creates an actionable ranking in which the first
recommended donors are not only medically compatible but also more likely to be reachable and responsive. The approach
can also provide an interpretable score breakdown so that an authorised user can understand why a donor was
prioritised.

## 7. Feasibility Study

**Technical feasibility.** The prototype can be developed using Python with Flask/FastAPI, a relational or NoSQL
database, and machine-learning libraries such as Scikit-learn. A weighted scoring model can be implemented first,
followed by Random Forest, XGBoost or another suitable classifier if sufficient training data are available. A web
dashboard can expose donor search, ranking and emergency-request functionality.

**Data feasibility.** Public, authorised, anonymised or synthetic donor data can be used for an academic prototype.
Relevant features may include blood group, donation frequency, donation interval, response history, availability
status and approximate location. Sensitive identifiers should not be used unnecessarily.

**Operational feasibility.** The system can operate as a recommendation and coordination layer for authorised
hospitals, blood banks or administrators. Users can submit an emergency request, apply compatibility constraints,
obtain a ranked list and contact selected donors through approved communication channels.

**Economic feasibility.** The prototype can be built with open-source software and local development environments,
avoiding public blockchain or proprietary infrastructure costs. Cloud deployment, SMS gateways and large-scale hosting
can be considered later if required.

**Privacy and security feasibility.** Because donor information is sensitive, the system should apply authentication,
role-based authorisation, encryption, data minimisation and secure API practices. Exact personal addresses and
unnecessary medical information should not be exposed in the recommendation interface.

## 8. Expected Outcomes

The expected result is a working prototype that accepts an emergency blood request and produces a ranked list of
compatible donor candidates. The ranking should combine basic compatibility with reliability and availability
indicators, allowing the system to distinguish between merely registered donors and donors who are more likely to
respond. The project should provide a donor dashboard, an emergency-request workflow, a recommendation module, and an
administrator or authorised-user interface. Evaluation can include Precision@K, Recall@K, ranking quality,
response-probability prediction metrics where labelled data are available, and the time taken to retrieve the top
candidates. The system should also provide an interpretable score or feature-contribution view to support transparent
decision-making.

## 9. Conclusion

Trust-aware recommendation can make emergency blood-donor discovery more practical by combining compatibility with the
factors that determine whether a donor is actually useful at the time of need. Recent research supports the use of
machine learning for predicting donor willingness, algorithmic matching for increasing donor action, location-based
matching for faster discovery, and real-time predictive systems for donor availability. The proposed project brings
these ideas together through an interpretable ranking framework centred on reliability and availability. A feasible
academic prototype can demonstrate how such a system may reduce search time, improve donor prioritisation and support
emergency blood coordination while maintaining privacy and leaving final medical eligibility decisions to authorised
healthcare personnel.

## References

1. H.-Y. Wu et al., "Predicting willingness to donate blood based on machine learning: two blood donor recruitments
   during COVID-19 outbreaks," *Scientific Reports*, vol. 12, 19165, 2022. DOI: 10.1038/s41598-022-21215-2.
2. S. Suessner et al., "Machine learning-based prediction of fainting during blood donations using donor properties
   and weather data as features," *BMC Medical Informatics and Decision Making*, vol. 22, 222, 2022.
   DOI: 10.1186/s12911-022-01971-x.
3. D. C. McElfresh et al., "Matching algorithms for blood donation," *Nature Machine Intelligence*, vol. 5,
   pp. 1108-1118, 2023. DOI: 10.1038/s42256-023-00722-5.
4. T. Kala, "A Location-Based Blood Donor Locator System Using Intelligent Matching Methodology," *International
   Journal of Innovative Research in Computer and Communication Engineering/Technology*, vol. 9, issue 4, 2023,
   pp. 112-118.
5. S. R. Akella, C. Thatiparthi, K. B. H. Mohammad, and L. D. Thammaneni, "A Machine Learning-Based Blood Donor
   Recommendation System to Enhance Blood Donation Efficiency," *Industrial Engineering Journal*, vol. 53, issue 5,
   2024.
6. "Real-Time Blood Donor Prediction and Campaign Optimization using DNN-Autoencoder and MERN Architecture," *2025 3rd
   International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS)*, 2025.
   DOI: 10.1109/ICSSAS66150.2025.11080959.
7. M. A. Badawi, "Artificial intelligence in blood donor management: A narrative review," *Vox Sanguinis*, first
   published online 28 October 2025, issue 2026. DOI: 10.1111/vox.70141.
