// import axios from 'axios';
// // import 'bootstrap/dist/css/bootstrap.min.css';
// import React from 'react';
// import './App.css';
// import Nav from './components/Nav';

// // function App() {
// //     return (
// //         <div className="App">
// //             <header className="App-header">
// //                 <img src={logo} className="App-logo" alt="logo" />
// //                 <p>
// //                     Edit <code>src/App.tsx</code> and save to reload!
// //                 </p>
// //                 <a
// //                     className="App-link"
// //                     href="https://reactjs.org"
// //                     target="_blank"
// //                     rel="noopener noreferrer"
// //                 >
// //                     Learn React
// //                 </a>
// //             </header>
// //         </div>
// //     );
// // }
// type Hello = {
//     Hello: string;
// };

// function App() {
//     const [data, setData] = React.useState<Hello>()
//     const url = `http://${process.env.REACT_APP_API_HOST}:3000/`
//     // console.log(url1);
//     // const url = 'http://172.21.65.108:3000/'
//     // console.log(url1, url)
//     const GetData = () => {
//         axios.get(url).then((res) => {
//             setData(res.data);
//         });
//     };
//     return (
//         <Nav />
//         <div>
//             <div>ここに処理を書いていきます</div>
//             {data ? <div>{data.Hello}</div> : <button onClick={GetData}>データを取得</button>}
//         </div>
//     );
// }

// export default App;


import 'bootstrap/dist/css/bootstrap.min.css';
import React from 'react';
import Editor from './components/Editor/index';
import Nav from './components/Nav';
type Item = {
    id: number
    title: string
}

const App: React.FC = () => {
    const items: Item[] = [
        {
            id: 1,
            title: '一番高い商品'
        }, {
            id: 2,
            title: '一番ださい商品'
        }
    ]

    return (
        <div className="App">
            <Nav />
            <Editor />

            {/* <Container>
                <Row>
                    <Col xs={9}>
                        <Editor />
                    </Col>
                    <Col>
                        param space
                    </Col>
                </Row>
            </Container> */}
            <div>ここに処理を書いていきます</div>
            <ul>
                {items.map((item: Item) => (
                    <li key={item.id}>{item.title}</li>
                ))}
            </ul>
        </div>
    );
}

export default App;