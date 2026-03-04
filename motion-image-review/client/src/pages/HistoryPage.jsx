import React, { useState } from "react";
import HistoryStats from "../components/HistoryStats";
import HistoryTable from "../components/HistoryTable";
import DetailModal from "../components/DetailModal";

export default function HistoryPage({ history }) {
  const [detailItem, setDetailItem] = useState(null);

  return (
    <div>
      {detailItem && <DetailModal item={detailItem} onClose={() => setDetailItem(null)} />}
      <HistoryStats history={history} />
      <HistoryTable history={history} onSelectItem={setDetailItem} />
    </div>
  );
}
